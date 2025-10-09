CREATE EXTERNAL LOCATION IF NOT EXISTS nlp_root
  URL 'abfss://nlp@<storage_account>.dfs.core.windows.net/'
  WITH (CREDENTIAL `cred_nlp_azmi`);
