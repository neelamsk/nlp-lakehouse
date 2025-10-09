-- External location -> catalog -> schemas -> grants
CREATE EXTERNAL LOCATION IF NOT EXISTS nlp_root
  URL 'abfss://nlp@<storage_account>.dfs.core.windows.net/'
  WITH (CREDENTIAL `cred_nlp_azmi`);

CREATE CATALOG IF NOT EXISTS nlp_dev
  MANAGED LOCATION 'abfss://nlp@<storage_account>.dfs.core.windows.net/uc/nlp_dev/';

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

GRANT USAGE ON CATALOG nlp_dev TO `account users`;
GRANT USAGE ON SCHEMA nlp_dev.bronze TO `account users`;
GRANT CREATE, SELECT, MODIFY ON SCHEMA nlp_dev.bronze TO `account users`;
GRANT USAGE ON SCHEMA nlp_dev.silver TO `account users`;
GRANT CREATE, SELECT, MODIFY ON SCHEMA nlp_dev.silver TO `account users`;
GRANT USAGE ON SCHEMA nlp_dev.gold TO `account users`;
GRANT CREATE, SELECT, MODIFY ON SCHEMA nlp_dev.gold TO `account users`;
GRANT USAGE ON SCHEMA nlp_dev.ml TO `account users`;
GRANT CREATE, SELECT, MODIFY ON SCHEMA nlp_dev.ml TO `account users`;
GRANT USAGE ON SCHEMA nlp_dev.agents TO `account users`;
GRANT CREATE, SELECT, MODIFY ON SCHEMA nlp_dev.agents TO `account users`;
