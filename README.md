# NLP Lakehouse (Azure Databricks + Unity Catalog + Delta on ADLS)

End-to-end NLP lakehouse on Azure Databricks that processes **millions of Amazon reviews** using **Unity Catalog**, **Delta Lake on ADLS Gen2**, **MLflow**, **dbt**, and a first-cut **Ops Agent** for model monitoring.

Built to be:
- Honest about scale: real multi-million row text dataset.
- Modern: Unity Catalog, external Delta on ADLS (no DBFS mounts).
- Governed: dbt Gold layer for analytics, tests, and documentation.
- Interview-ready: clear phases, clean repo layout, and repeatable patterns.

---

## 1. What this project is now (Final Snapshot)

### 🧱 Data & Lakehouse

**Storage**

- ADLS Gen2 account with container: `nlp`
- Folder layout inside container:
  - `raw/`
  - `bronze/`
  - `silver/`
  - `gold/`
  - `ml/`
  - `agents/`

**Compute & Governance**

- Azure Databricks workspace
- Unity Catalog:
  - Catalog: `nlp_databricks`
  - Schema: `nlp_dev`
    - `nlp_dev.bronze`
    - `nlp_dev.silver`
    - `nlp_dev.gold` (plus `gold_gold` for dbt outputs)
    - `nlp_dev.ml`
    - `nlp_dev.agents`

All core tables are **Delta** tables, backed by **abfss://** paths in ADLS.



## Architecture (high level)
**Azure**: *Databricks (Workflows/Jobs, MLflow, UC) + ADLS Gen2 (HNS) + Key Vault + (optional) SQL Warehouse for dbt/BI.*

**Data flow**: Public text dataset → **Bronze** (immutable) → clean/normalize in **Silver** → feature & inference outputs into **Gold** → **dbt** builds analytics marts → BI charts + README KPIs → **agent** summarizes runs, flags drift/cost, and recommends action.

```
          +----------------+      +--------------------+
Public -> |   Ingest       |  ->  |   Bronze (Delta)   |
Dataset   | (Autoloader)   |      +--------------------+
          +----------------+                 |
                                            \|/
                                   +--------------------+
                                   |  Silver (Delta)    |  Clean text, normalize
                                   +--------------------+
                                            \|/
                                   +--------------------+
                                   |  Features & Model  |  TF‑IDF + LR baseline
                                   +--------------------+  (MLflow tracking)
                                            \|/
                                   +--------------------+
                                   |  Gold (Delta)      |  predictions, perf, cost
                                   +--------------------+
                                    /         |          \
                                   /          |           \
                            (dbt models)   (BI KPI)      (Agent: drift/cost)
```

---

## 2. Data Flow & Tables

### 📥 Raw → Bronze

**Source**

- fastText-format **Amazon Reviews** text:
  - `__label__X` followed by free text review
  - Multi-million rows (sliced for experimentation)

**Notebook: `01_ingest_bronze`**

- Reads raw text files from:
  - `abfss://nlp@<storage>.dfs.core.windows.net/raw/amazon_reviews/...`
- Parses:
  - `label` from `__label__X`
  - `text` from remainder of the line
- Writes Delta table:

```sql
nlp_databricks.nlp_dev.bronze.fasttext_bronze
```

> Bronze = minimally processed, schema-on-read turned into schema-on-write.

---

### 🧼 Bronze → Silver

**Notebook: `02_clean_silver`**

- Normalizes text:
  - Lowercase
  - Punctuation removal
  - Whitespace cleanup
- Adds simple features:
  - `word_count`
  - `char_count`
- Filters out:
  - Reviews that are too short
  - Reviews that are too long / clearly noisy
- Writes cleaned Delta table:

```sql
nlp_databricks.nlp_dev.silver.fasttext_silver
```

- Runs:
  - `OPTIMIZE` + `ZORDER` (on relevant columns) to improve performance

> Silver = cleaned, filtered, and lightly featured text ready for modeling.

---

### 🪙 Silver → Gold (Features + Splits)

**Notebook: `03_gold_features`**

- Reads:

```sql
SELECT * FROM nlp_databricks.nlp_dev.silver.fasttext_silver
```

- Processing:
  - Tokenizes text
  - Computes TF-IDF features into a `tfidf_features` vector column (Spark ML)
  - Encodes labels into numeric form
  - Adds `split` column with values:
    - `train`
    - `val`
    - `test`

- Writes feature table:

```sql
nlp_databricks.nlp_dev.gold.fasttext_gold
```

> Gold (feature layer) = single source of truth for train/val/test splits.

---

## 3. Model & MLflow

### 🧠 Baseline Model: Logistic Regression

**Notebook: `04_train_lr_baseline`**

- Reads Gold feature table
- Splits into:
  - `df_train`, `df_val`, `df_test` (by `split` column)
- Trains a **Logistic Regression classifier** using **Spark ML** on TF-IDF vectors
- Evaluates:
  - Accuracy
  - F1 (overall and per label where helpful)
- Tracks with **MLflow**:
  - Parameters (regularization, features, etc.)
  - Metrics (accuracy, F1)
  - Model artifact

**Model registry**

- Registers model in Unity Catalog:

```sql
nlp_databricks.nlp_dev.ml.fasttext_sentiment_lr
```

Model is versioned and ready to serve / batch-score.

---

### 📦 Batch Inference

**Notebook: `05_batch_inference`**

- Loads latest `fasttext_sentiment_lr` version from UC model registry
- Runs batch inference on:
  - Full `fasttext_gold` table, or
  - A chosen slice for experimentation
- Writes predictions to Delta table:

```sql
nlp_databricks.nlp_dev.ml.fasttext_predictions
```

Columns typically include:
- `text`
- `true_label`
- `pred_label`
- Prediction confidence / probability vector
- Inference timestamps

> This table is the bridge between ML and analytics: dbt uses it as a source.

---

## 4. Analytics & dbt Layer

This lives in a **separate repo**:

- Repo name (example): `nlp_lakehouse_analytics`
- Engine: **dbt** with `dbt-databricks`

### 🔌 Sources

`models/sources.yml` defines:

```yml
sources:
  - name: ml
    database: nlp_databricks
    schema: nlp_dev.ml
    tables:
      - name: fasttext_predictions
```

So dbt sees:

```sql
source('ml', 'fasttext_predictions') -> nlp_databricks.nlp_dev.ml.fasttext_predictions
```

---

### 🧮 Models

#### 1) `fact_sentiment_confusion.sql`

- Aggregates predictions into a confusion matrix:

```sql
SELECT
  true_label,
  pred_label,
  COUNT(*) AS review_count
FROM {{ source('ml', 'fasttext_predictions') }}
GROUP BY true_label, pred_label
```

- Materializes as:

```sql
nlp_databricks.nlp_dev.gold_gold.fact_sentiment_confusion
```

#### 2) `fact_sentiment_summary.sql`

- Uses CTEs:
  - `base`
  - `per_class`
  - `overall`
- Computes:
  - Per-class accuracy
  - Overall accuracy
- Materializes as:

```sql
nlp_databricks.nlp_dev.gold_gold.fact_sentiment_summary
```

### ✅ Tests

`schema.yml` includes:

- `not_null` tests on key columns:
  - `true_label`
  - `pred_label`
  - `review_count`
  - Accuracy columns

`dbt test` → all current tests passing.

> dbt represents the **governed analytics layer** on top of ML outputs: repeatable SQL, version control, and tests.

---

## 5. Agent Layer (Ops Copilot v0)

Agent logic lives in the **`nlp_dev.agents`** schema.

### 📈 `agents.model_metrics`

**Notebook: `06_agent_metrics`**

- Reads:
  - MLflow metrics
  - dbt summary / confusion tables
- Stores rows in:

```sql
nlp_databricks.nlp_dev.agents.model_metrics
```

Schema (conceptually):

- `model_name`
- `model_version`
- `run_id`
- `metric_scope` (e.g., `val`, `test`)
- `overall_accuracy`
- Per-label metrics
- `created_at`

> This acts as a time-series log of model performance across runs.

---

### 🧠 `agents.ops_reports`

**Notebook: `07_agent_ops_report`**

- Reads:
  - Latest `agents.model_metrics`
  - Gold analytics tables (`fact_sentiment_confusion`, `fact_sentiment_summary`)
- Builds a context dict
- Passes it through a small **rule-based “agent brain”** (`build_ops_summary`) that:
  - Classifies performance tier (`excellent`, `good`, `concerning`)
  - Sets `needs_retrain` flag
  - Generates a human-readable `summary_text`
  - Produces a `flags` dict (e.g., classes with low accuracy)

- Writes a row into:

```sql
nlp_databricks.nlp_dev.agents.ops_reports
```

Typical columns:

- `report_id`
- `generated_at`
- `model_name`, `model_version`, `run_id`
- `overall_accuracy`
- `summary_text`
- `flags_json`

> This is an **agent-ready monitoring layer**: you already have an observe → decide → log loop. Swapping the rule engine for an LLM later is straightforward.

---

### 📊 `agents.prediction_stats` (Drift foundation)

**Notebook: `08_agent_prediction_stats`**

- Aggregates prediction stats:
  - By date
  - By label
  - By predicted label
- Writes to:

```sql
nlp_databricks.nlp_dev.agents.prediction_stats
```

This table is the foundation for:
- Drift analysis
- Stability and volume monitoring
- Future agent rules/LLM prompts around drift & cost.

---

## 6. Repos & Structure

### Repo 1: `nlp-lakehouse` (Databricks / ML / Agents)

Suggested layout:

```text
nlp-lakehouse/
  README.md
  databricks/
    01_ingest_bronze
    02_clean_silver
    03_gold_features
    04_train_lr_baseline
    05_batch_inference
    06_agent_metrics
    07_agent_ops_report
    08_agent_prediction_stats
  docs/
    architecture.md
    screenshots/
      mlflow_runs.png
      uc_model_registry.png
      delta_tables.png
      agent_ops_report_sample.png
```

### Repo 2: `nlp_lakehouse_analytics` (dbt)

```text
nlp_lakehouse_analytics/
  README.md
  dbt_project.yml
  models/
    sources.yml
    schema.yml
    fact_sentiment_confusion.sql
    fact_sentiment_summary.sql
  .gitignore             # exclude profiles.yml
```

---

## 7. How to Run (High-level)

1. **Prereqs**
   - Databricks workspace with Unity Catalog enabled
   - UC catalog `nlp_databricks` and schema `nlp_dev`
   - ADLS Gen2 container `nlp` wired as an external location
   - Cluster with:
     - Auto-terminate (10–15 min)
     - 1–2 small workers

2. **Ingest**
   - Upload fastText Amazon reviews into:
     - `raw/amazon_reviews/`
   - Run `01_ingest_bronze` → creates `bronze.fasttext_bronze`

3. **Clean**
   - Run `02_clean_silver` → creates `silver.fasttext_silver`

4. **Features & Splits**
   - Run `03_gold_features` → creates `gold.fasttext_gold`

5. **Train & Register Model**
   - Run `04_train_lr_baseline` → MLflow run + UC model registered

6. **Batch Inference**
   - Run `05_batch_inference` → writes `ml.fasttext_predictions`

7. **dbt Analytics**
   - In `nlp_lakehouse_analytics`:
     - `dbt debug`
     - `dbt run`
     - `dbt test`
   - Confusion and summary tables created in `gold_gold` schema

8. **Agent Metrics & Reports**
   - Run `06_agent_metrics` → populate `agents.model_metrics`
   - Run `07_agent_ops_report` → populate `agents.ops_reports`
   - Run `08_agent_prediction_stats` (optional) → populate `agents.prediction_stats`

---

## 8. Cost & Ops Notes

- Use **Jobs** clusters or small interactive clusters with:
  - Auto-terminate
  - Minimal workers
- Only **OPTIMIZE** on frequently-queried tables; avoid blanket Z-ORDER.
- Keep any **SQL Warehouse** stopped if you’re not actively using it.
- Subscription-level **budget + alerts** recommended if this runs regularly.

---

## 9. How to talk about this in interviews

Example one-liner:

> “I built an NLP lakehouse on Azure Databricks with Unity Catalog, processing multi-million Amazon reviews into Delta tables on ADLS, trained a TF-IDF + Logistic Regression baseline with MLflow tracking, layered dbt marts on top, and added an Ops Agent table that logs model performance and retrain recommendations.”

Key talking points:

- UC + external locations, **no DBFS mounts**
- Clear **Bronze / Silver / Gold / ML / Agents** separation
- MLflow + UC model registry usage
- dbt tests and confusion/summary marts
- Agent pattern: `model_metrics` → `ops_reports` (observe → decide → log)
- Ready for future:
  - LLM-powered Ops Copilot
  - DistilBERT vs LR comparison
  - Drift and cost optimization logic

---

## 10. Roadmap / Stretch

- Add **LLM-based Ops Copilot** (Azure OpenAI) using the existing `agents.*` tables as context.
- Add **drift detection** (PSI / KL) using `agents.prediction_stats`.
- Compare **DistilBERT** vs LR:
  - Accuracy vs cost/1k predictions.
- Add **CI/CD** with Databricks Asset Bundles + GitHub Actions.

---

## License

MIT or similar (to be decided).



