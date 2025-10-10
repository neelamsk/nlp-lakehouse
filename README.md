# NLP Lakehouse (Azure Databricks + Unity Catalog + Delta on ADLS)

> **Goal:** An interview‑ready, cost‑aware NLP project that processes **1–5M+ documents** end‑to‑end on Azure Databricks with **Unity Catalog (UC)**, **Delta Lake** on ADLS Gen2, **MLflow** model tracking/registry, **dbt** Gold marts, and a small **Ops Copilot** agent for drift/cost insights. Built for portfolio credibility and repeatability.

---

## Why this project?
- **Real scale:** Truthfully say you processed *millions* of texts.
- **Modern lakehouse:** UC governance + Delta tables on ADLS (no lock‑in, no DBFS mounts).
- **Ops maturity:** MLflow registry, drift checks (PSI/KL), cost metrics, and agent recommendations.
- **Analytics storytelling:** dbt marts + dashboard screenshots (latency, accuracy, drift, cost/1k preds).

---

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

> **Note:** We will use **abfss://** URIs to read/write ADLS paths; **no DBFS mounts**. UC provides the catalog/schemas and secure external locations.

---

## Tech Stack
- **Compute**: Azure Databricks (Workflows/Jobs, MLflow, UC)
- **Storage**: ADLS Gen2 + Delta Lake (ACID, time travel, OPTIMIZE/Z‑ORDER later)
- **Governance**: Unity Catalog (catalog/schema/tables, external locations, grants)
- **Secrets**: Azure Key Vault (Databricks secret scope)
- **Analytics**: dbt (dbt‑databricks) → optional SQL Warehouse (stopped by default), Power BI/DB SQL dashboards
- **Models**: Baseline TF‑IDF + Logistic Regression; stretch: DistilBERT
- **Agent**: Small LLM (Azure OpenAI “small/mini”) for Ops Copilot (summaries, drift/cost hints)

---

## Project Phases (with acceptance checks)
- **Phase 0 – Foundations** (this README, infra wiring)
  - 0.1 New subscription + RG + budget ✅
  - 0.2 ADLS Gen2 + `nlp` container + `/raw /silver /gold /ml /agents /scratch` ✅
  - 0.3 Databricks + **Unity Catalog** (Access Connector → Storage Credential → External Location → Catalog/ Schemas) ☐
  - 0.4 Key Vault + secret scope + `dbutils.secrets.get` smoke test ☐
  - 0.5 Cluster policy (auto‑terminate 10 min, tiny nodes, autoscale 1‑2) ☐
  - 0.6 MLflow smoke test (log param/metric/artifact) ☐
- **Phase 1 – Ingest & Explore**
  - Choose dataset (e.g., **Amazon Reviews JSONL**, or **AG News**); land to `/raw` ☐
  - Autoloader to **Bronze Delta**, schema inference, incremental ☐
  - EDA (lengths, nulls, class balance) ☐
- **Phase 2 – Clean & Persist (Silver)**
  - Text cleaning (lowercase, punct, stopwords; optional lemmatization) ☐
  - Partition by date; write Silver ☐
- **Phase 3 – Features & Baseline**
  - TF‑IDF + Logistic Regression (or LightGBM) ☐
  - MLflow log + register; batch inference → predictions table ☐
- **Phase 4 – Gold & dbt**
  - Build marts: `reviews_daily`, `model_perf_daily`, `inference_latency_daily`, `agent_actions_log` ☐
- **Phase 5 – Agent (Ops Copilot)**
  - Summarize latest runs; flag imbalance/drift; recommend next steps ☐
- **Phase 6 – Scale & Performance**
  - OPTIMIZE/Z‑ORDER (targeted), caching, broadcast joins; show before/after ☐
- **Phase 7 – Production Metrics**
  - Persist p50/p95 latency, cost/1k preds, drift days, agent response time ☐
- **Phase 8 – Cost Optimizer & Auto‑Retrain**
  - Drift stats; accuracy vs cost trade‑off; propose cheaper model if Δacc < 2% ☐
- **Phase 9 – Observability & Docs**
  - Dashboards & screenshots; curated README KPIs ☐
- **Phase 10 – Stretch**
  - Real **Kafka** streaming (self‑managed) ☐
  - **DistilBERT** accuracy vs cost demo ☐
  - **RAG Copilot** for FAQs in a subset class ☐

---

## Repository Layout (evolving)
> You’re starting with just this folder. We’ll add folders as we build.
```
nlp-lakehouse/
  README.md                 <-- this file
  databricks/               <-- notebooks, SQL, jobs (added later)
  src/                      <-- python libs (drift, cost, utils)
  dbt/                      <-- models, tests, docs (analytics)
  infra/                    <-- terraform/bicep (optional, later)
  docs/                     <-- arch diagram, KPIs, screenshots
```

---

## Step‑by‑Step Setup (Concise Runbook)

### 1) Unity Catalog wiring (after workspace is created)
1. **Create Access Connector** (Azure Databricks Access Connector) in the same RG/region.  
2. On the storage account (e.g., `nlplakeadls001`), grant the connector **Storage Blob Data Contributor** (IAM).  
3. In Databricks **Account Console / Data**:
   - **Storage Credential**: type *Azure managed identity*, pick the Access Connector.  
   - **External Location**: `abfss://nlp@<account>.dfs.core.windows.net/` using that credential.  
   - **Metastore**: create (East US 2) if needed and **attach** workspace.  
   - **Catalog**: `nlp_dev`.  
   - **Schemas** with managed locations (optional but tidy):  
     - `nlp_dev.bronze` → `abfss://nlp@<account>.dfs.core.windows.net/bronze/`  
     - `nlp_dev.silver` → `.../silver/`, etc.

**SQL smoke test**
```sql
USE CATALOG nlp_dev;
CREATE SCHEMA IF NOT EXISTS bronze;
USE SCHEMA bronze;

CREATE TABLE IF NOT EXISTS test_delta (id INT, txt STRING) USING DELTA;
INSERT INTO test_delta VALUES (1, 'hello uc');
SELECT * FROM test_delta;
```

### Credential strategy: workspace default vs. managed identity (recommended)

**Two ways UC reaches storage:**
- **Workspace default credential** (Databricks-managed): quick for PoC; scope-limited to a Databricks path (e.g., `...unity-catalog-storage@dbstorage...`). Not ideal for real projects.
- **Storage Credential via Azure Databricks Access Connector (Managed Identity)** ← **Recommended**
  - We create a Storage Credential (e.g., `cred_nlp_azmi`) that uses the Access Connector (Resource ID).
  - We create an External Location (e.g., `nlp_root`) that points to **our** ADLS Gen2 container/prefix.
  - Each schema gets a **managed location** matching our layers (`/bronze`, `/silver`, `/gold`, `/ml`, `/agents`).

**Why this is best practice:** least-privilege RBAC/ACLs on *our* storage, portable data (no vendor path), clean folder layout, easy ops & compliance.


### 2) Secrets (Key Vault + Databricks scope)
- Create **Key Vault**, enable purge protection/soft delete.  
- Add secrets you’ll need (OpenAI key later, optional service principal).  
- In Databricks, create a **secret scope** (Key Vault‑backed preferred) and test:
```python
dbutils.secrets.get(scope="kv-nlp", key="openai-api-key")  # example later
```

### 3) Cluster policy (guardrail)
- **Auto‑terminate 10 min**, **autoscale 1–2 workers**, small node (e.g., `Standard_DS3_v2` or smallest available), **spot workers** allowed for dev.  
- Use **Jobs** compute rather than interactive clusters for pipelines.

### 4) MLflow smoke test
```python
import mlflow
with mlflow.start_run():
    mlflow.log_param("hello", "world")
    mlflow.log_metric("accuracy", 0.99)
# Verify in Experiments UI
```

---

## Ingestion & Bronze (Autoloader)
Example (JSONL reviews) – replace path with your dataset:
```python
from pyspark.sql.functions import input_file_name, current_timestamp

raw_path = "abfss://nlp@<account>.dfs.core.windows.net/raw/amazon_reviews/"
bronze_path = "abfss://nlp@<account>.dfs.core.windows.net/bronze/amazon_reviews/"

df = (spark.readStream
      .format("cloudFiles")
      .option("cloudFiles.format", "json")
      .option("cloudFiles.inferColumnTypes", "true")
      .load(raw_path))

df = df.withColumn("_ingested_at", current_timestamp()) \
       .withColumn("_source_file", input_file_name())

(df.writeStream
   .format("delta")
   .option("checkpointLocation", bronze_path + "_checkpoint")
   .outputMode("append")
   .start(bronze_path))
```
- Use **`cloudFiles`** (Autoloader) for incremental discovery & schema evolution.
- For **batch**, replace `readStream`/`writeStream` with `read`/`write`.

---

## Silver Cleaning (example)
```python
from pyspark.sql import functions as F

bronze = "abfss://nlp@<account>.dfs.core.windows.net/bronze/amazon_reviews/"
silver = "abfss://nlp@<account>.dfs.core.windows.net/silver/amazon_reviews/"

df = spark.read.format("delta").load(bronze)

clean = (df
  .withColumn("text", F.lower(F.col("reviewText")))
  .withColumn("text", F.regexp_replace("text", r"[^\w\s]", " "))
  .filter(F.length("text") > 20))

(clean.write
   .format("delta")
   .mode("overwrite")
   .option("overwriteSchema", "true")
   .save(silver))
```

---

## Baseline Features & Model (MLflow)
```python
import mlflow, mlflow.sklearn
from pyspark.sql.functions import col
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

pdf = spark.read.format("delta").load(silver).select("text","label").dropna().toPandas()
X_train, X_test, y_train, y_test = train_test_split(pdf.text, pdf.label, test_size=0.2, random_state=42)

pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=200000)),
                 ("lr", LogisticRegression(max_iter=200))])

with mlflow.start_run():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_weighted", f1)
    mlflow.sklearn.log_model(pipe, "model", registered_model_name="nlp_baseline_tfidf_lr")
```

---

## Gold & dbt (marts)
**Target marts:**
- `reviews_daily` (counts, avg length)
- `model_perf_daily` (accuracy, F1, loss)
- `inference_latency_daily` (p50/p95 latency)
- `agent_actions_log` (recommended vs applied, outcome)

> Keep your **SQL Warehouse OFF** when not querying to control cost.

---

## Ops Copilot (small agent)
- Pull latest MLflow runs (metrics, tags).  
- Check **class imbalance** and **drift** (PSI/KL) from Silver→Gold stats.  
- Draft **next steps**: retrain, adjust batch size, consider cheaper model if Δacc < 2%.  
- Persist outputs to `gold.agent_actions_log` for BI.

---

## KPIs to publish in README (replace with your real numbers)
- **Docs processed:** 5.3M  
- **Baseline:** TF‑IDF + LR → **Accuracy 92.1% / F1 0.91**  
- **Batch ingest p95 latency:** 7.8 min / day partition  
- **Cost/1k predictions:** \$0.0X (VM + DBU calc)  
- **Drift detection days:** 6/90 days (6.7%)  
- **Agent recommendations applied:** 12, **80%** success

---

## Cost Controls
- **Jobs clusters** only, **auto‑terminate 10 min**, **autoscale 1–2**, **small nodes**, **spot workers** for dev.  
- **Optimize** only hot partitions; avoid blanket Z‑ORDER.  
- Keep **SQL Warehouse stopped**.  
- Azure **Budgets + alerts** at subscription/RG scope.

---

## Security & Governance
- **RBAC**: Storage Blob Data Contributor to the **Access Connector** only.  
- **ACLs**: root container + default ACLs for inheritance.  
- **Unity Catalog**: grants on catalog/schemas; optional row/column controls.  
- **Secrets**: Key Vault–backed secret scope; no creds in code.  
- **Audit**: enable storage **change feed** and soft delete; keep MLflow run logs.

---

## CI/CD & Automation (later)
- **GitHub Actions**:
  - Lint & unit tests → build → **databricks CLI** or **Databricks Asset Bundles** to deploy jobs/notebooks.  
  - **dbt**: run tests, build docs, publish artifacts (screenshots to `/docs`).  
- **Terraform** (optional): UC objects (catalog, schemas, credentials, external locations), secret scopes, cluster policies.

---

## Resume / Portfolio snippet
- “Built an Azure Databricks + Unity Catalog NLP lakehouse (Delta on ADLS). Processed **5M+** documents, logged models in MLflow, added dbt Gold marts & a small Ops Copilot for drift/cost actions. **Accuracy 92%**, **cost/1k preds –38%**, **p95 batch latency 7.8 min**.”

---

## Glossary (quick)
- **Unity Catalog (UC):** Cross‑workspace governance & catalog for Databricks.  
- **Hive Metastore:** Legacy per‑workspace metastore.  
- **DBFS:** Databricks virtual filesystem; we **don’t** mount ADLS here.  
- **ABFS/abfss://:** Connector/URI for ADLS Gen2 (TLS with `abfss://`).  
- **Delta Lake / Delta table:** Parquet files + `_delta_log` = ACID + time travel.  
- **MLflow:** Experiment tracking, model registry.  
- **Autoloader:** Incremental file discovery for streaming/batch ingress.

---

**Troubleshooting – “wrong credential (nlp_databricks) used”**
- **Symptom:**  
  `PERMISSION_DENIED: credential 'nlp_databricks' ... only allowed to access ... unity-catalog-storage@dbstorage...`
- **Cause:** External Location was created with the workspace default credential, so UC uses that instead of our MI.
- **Fix (what we did):**
  1. Create **Storage Credential** with the **Access Connector (Resource ID)** → `cred_nlp_azmi`.
  2. Create **External Location** → `nlp_root` using `cred_nlp_azmi`.
  3. Grant UC access:  
     ```sql
     GRANT READ FILES, WRITE FILES ON EXTERNAL LOCATION nlp_root TO `account users`; 
     -- or to your UPN from: SELECT current_user();
     ```
     *(Metastore privilege v1.0 doesn’t need/allow `USAGE` on storage credential.)*
  4. If an old External Location was tied to the default credential and blocked by dependencies, **drop schemas/tables** or **Force delete** the location, then recreate it with `cred_nlp_azmi`.
  5. Ensure schema **managed locations** map to folders (do this at create time if ALTER isn’t supported):
     ```sql
     CREATE SCHEMA IF NOT EXISTS nlp_dev.bronze
       MANAGED LOCATION 'abfss://nlp@<storage_account>.dfs.core.windows.net/bronze/';
     ```
  6. Verify a table’s physical location:  
     ```sql
     DESCRIBE DETAIL nlp_dev.bronze.some_table;  -- see 'location'
     ```


---

## Roadmap / TODO
- [ ] Finish Phase 0 (UC + secret scope + cluster policy + MLflow smoke)  
- [ ] Land dataset in `/raw` and wire **Autoloader** to **Bronze**  
- [ ] Silver cleaning (text normalize)  
- [ ] Baseline TF‑IDF + LR + MLflow registry  
- [ ] Batch inference → predictions (Gold)  
- [ ] dbt marts + dashboard screenshots  
- [ ] Ops Copilot agent (drift/cost)  
- [ ] Performance pass (OPTIMIZE/Z‑ORDER selectively)  
- [ ] Cost optimizer + auto‑retrain proposal  
- [ ] Stretch: Kafka streaming + DistilBERT + RAG

---

## License
Choose one (e.g., MIT).
