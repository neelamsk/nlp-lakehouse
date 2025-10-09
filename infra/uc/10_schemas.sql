CREATE SCHEMA IF NOT EXISTS nlp_dev.bronze
  MANAGED LOCATION 'abfss://nlp@<storage_account>.dfs.core.windows.net/bronze/';
CREATE SCHEMA IF NOT EXISTS nlp_dev.silver
  MANAGED LOCATION 'abfss://nlp@<storage_account>.dfs.core.windows.net/silver/';
CREATE SCHEMA IF NOT EXISTS nlp_dev.gold
  MANAGED LOCATION 'abfss://nlp@<storage_account>.dfs.core.windows.net/gold/';
CREATE SCHEMA IF NOT EXISTS nlp_dev.ml
  MANAGED LOCATION 'abfss://nlp@<storage_account>.dfs.core.windows.net/ml/';
CREATE SCHEMA IF NOT EXISTS nlp_dev.agents
  MANAGED LOCATION 'abfss://nlp@<storage_account>.dfs.core.windows.net/agents/';
