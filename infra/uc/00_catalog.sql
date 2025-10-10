CREATE CATALOG IF NOT EXISTS nlp_dev
  MANAGED LOCATION 'abfss://nlp@<storage_account>.dfs.core.windows.net/uc/nlp_dev/';
-- <storage_account> with Actual ADLS account while executing the script
