databricks auth login --host https://dbc-6efc42df-8580.cloud.databricks.com --profile bootcamp
export DATABRICKS_CONFIG_PROFILE=bootcamp
databricks bundle validate && databricks bundle validate -t staging