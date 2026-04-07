# Databricks notebook source
# MAGIC %md
# MAGIC # Log and Register Agent
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Evaluates the agent
# MAGIC 2. Logs the model to MLflow
# MAGIC 3. Registers it to Unity Catalog
# MAGIC 4. Sets the model version for downstream tasks

# COMMAND ----------


import mlflow
from databricks.sdk.runtime import dbutils
from loguru import logger

from arxiv_curator.agent import ArxivAgent
from arxiv_curator.config import load_config
from arxiv_curator.evaluation import (
    hook_in_post_guideline,
    polite_tone_guideline,
    word_count_check,
)

# Get parameters from workflow (passed via base_parameters in job YAML)
env = dbutils.widgets.get("env")
git_sha = dbutils.widgets.get("git_sha")
run_id = dbutils.widgets.get("run_id")

# Load configuration
cfg = load_config("project_config.yml", env=env)

model_name = f"{cfg.catalog}.{cfg.schema}.arxiv_agent"

logger.info(f"Environment: {env}")
logger.info(f"Git SHA: {git_sha}")
logger.info(f"Run ID: {run_id}")
logger.info(f"Model Name: {model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Agent and Run Evaluation

# COMMAND ----------

# Initialize the agent
mlflow.set_experiment(cfg.experiment_name)

agent = ArxivAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=cfg.system_prompt,
    catalog=cfg.catalog,
    schema=cfg.schema,
    genie_space_id=cfg.genie_space_id,
)

# Load evaluation inputs
# Use dbutils to get the notebook path and construct the file path
notebook_path = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
bundle_root = "/".join(
    notebook_path.split("/")[:-3]
)  # Go up 3 levels from resources/deployment_scripts to bundle root
eval_file_path = f"/Workspace{bundle_root}/files/eval_inputs.txt"

with open(eval_file_path) as f:
    eval_data = [{"inputs": {"question": line.strip()}} for line in f if line.strip()]


def predict_fn(question: str) -> str:
    """Predict function that wraps the agent for evaluation."""
    request = {"input": [{"role": "user", "content": question}]}
    result = agent.predict(request)
    return result.output[-1].content


# Run evaluation
results = mlflow.genai.evaluate(
    predict_fn=predict_fn,
    data=eval_data,
    scorers=[word_count_check, polite_tone_guideline, hook_in_post_guideline],
)

logger.info("\n✓ Evaluation complete!")
logger.info(f"  Metrics: {results.metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log and Register Model

# COMMAND ----------

import random
from datetime import datetime

from mlflow import MlflowClient
from mlflow.models.resources import (
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksTable,
    DatabricksVectorSearchIndex,
)

# Define resources
resources = [
    DatabricksServingEndpoint(endpoint_name=cfg.llm_endpoint),
    DatabricksGenieSpace(genie_space_id=cfg.genie_space_id),
    DatabricksVectorSearchIndex(index_name=f"{cfg.catalog}.{cfg.schema}.arxiv_index"),
    DatabricksTable(table_name=f"{cfg.catalog}.{cfg.schema}.arxiv_papers"),
    DatabricksSQLWarehouse(warehouse_id=cfg.warehouse_id),
    DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
]

# Create test request
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

test_request = {
    "input": [{"role": "user", "content": "What are recent papers about LLMs and reasoning?"}],
    "custom_inputs": {
        "session_id": session_id,
        "request_id": request_id,
    },
}

model_config = {
    "catalog": cfg.catalog,
    "schema": cfg.schema,
    "genie_space_id": cfg.genie_space_id,
    "system_prompt": cfg.system_prompt,
    "llm_endpoint": cfg.llm_endpoint,
}

# Log model
ts = datetime.now().strftime("%Y-%m-%d")
with mlflow.start_run(
    run_name=f"arxiv-agent-{ts}", tags={"git_sha": git_sha, "run_id": run_id}
) as run:
    model_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="agent.py",
        resources=resources,
        input_example=test_request,
        model_config=model_config,
    )
    mlflow.log_metrics(results.metrics)

# Register model
registered_model = mlflow.register_model(
    model_uri=model_info.model_uri, name=model_name, env_pack="databricks_model_serving"
)

# Set alias
client = MlflowClient()
client.set_registered_model_alias(
    name=model_name,
    alias="latest-model",
    version=registered_model.version,
)

logger.info("\n✓ Model registered!")
logger.info(f"  Name: {registered_model.name}")
logger.info(f"  Version: {registered_model.version}")

# Set task value for downstream tasks
dbutils.jobs.taskValues.set(key="model_version", value=registered_model.version)
