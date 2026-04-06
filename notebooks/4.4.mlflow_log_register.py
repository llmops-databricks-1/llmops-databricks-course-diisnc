# Databricks notebook source
import random
from datetime import datetime

import mlflow
from mlflow import MlflowClient
from mlflow.models.resources import (
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,  # for genie
    DatabricksTable,  # for genie
    DatabricksVectorSearchIndex,
)

# ideally, lakebase would also have its own resource
from valuation_curator.agent import ValuationAgent
from valuation_curator.config import ProjectConfig
from valuation_curator.evaluation import (
    evidence_citations_scorer,
    hook_in_post_guideline,
    mentions_valuation_docs,
    professional_audit_tone_guideline,
    uses_detect_anomalies_tool,
    word_count_check,
)

# COMMAND ----------
# Initialize the agent
# NOTE: THIS NB ONLY RUNS IN DATABRICKS
cfg = ProjectConfig.from_yaml("../project_config.yml")
mlflow.set_experiment(cfg.experiment_name)

agent = ValuationAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=cfg.system_prompt,
    catalog=cfg.catalog,
    schema=cfg.schema,
    genie_space_id=cfg.genie_space_id,
    lakebase_project_id=cfg.lakebase_project_id,
)

# COMMAND ----------
# Load evaluation inputs (eval_inputs.txt)
with open("../eval_inputs.txt") as f:
    eval_data = [{"inputs": {"question": line.strip()}} for line in f if line.strip()]


def predict_fn(question: str) -> str:
    """Predict function that wraps the agent for evaluation."""
    request = {"input": [{"role": "user", "content": question}]}
    result = agent.predict(request)
    return result.output[-1].content


# COMMAND ----------
# Run evaluation
# on the output of this cell, when we click on a trace ID, we can add feedback saying that
# we do not agree with a certain classification. This is valid when we have a judge, not
# when we have a guideline which is the case for word_count_check, hook_in_post_guideline,
# and professional_audit_tone_guideline

results = mlflow.genai.evaluate(
    predict_fn=predict_fn,
    data=eval_data,
    scorers=[
        word_count_check,
        professional_audit_tone_guideline,
        hook_in_post_guideline,
        mentions_valuation_docs,
        evidence_citations_scorer,
        uses_detect_anomalies_tool,
    ],
)

# COMMAND ----------
# MAGIC %md
# MAGIC After evaluation, to log the model, we need a separate file valuation_agent.py
# MAGIC where we basically initialize the agent. Instead of having the whole agent logic
# MAGIC in the valuation_agent.py, it's easier for testing if the agent has its own file,
# MAGIC and on valuation_agent.py we just initialize it.

# COMMAND ----------
# from mlflow.models.resources
# databricks will create a service principle for me with permission for these objects and
# the agent will be able to use these tools.
resources = [
    DatabricksServingEndpoint(endpoint_name=cfg.llm_endpoint),
    DatabricksGenieSpace(genie_space_id=cfg.genie_space_id),
    DatabricksVectorSearchIndex(index_name=f"{cfg.catalog}.{cfg.schema}.valuation_index"),
    DatabricksTable(table_name=f"{cfg.catalog}.{cfg.schema}.customs_valuation_metadata"),
    DatabricksSQLWarehouse(warehouse_id=cfg.warehouse_id),
    DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
    # another db serving endpoint for embedding model for vector search index
]

# COMMAND ----------
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

# required: to log a model we need to provide test example so it can generate its
# signature
test_request = {
    "input": [
        {"role": "user", "content": "In which documents do I have a 3.5% royalty?"}
    ],
    "custom_inputs": {
        "session_id": session_id,
        "request_id": request_id,
    },
}

# actual config to overwrite the default values in the valuation_agent.py
# taking system_prompt from config file with the prompt to evaluate
model_config = {
    "catalog": cfg.catalog,
    "schema": cfg.schema,
    "genie_space_id": cfg.genie_space_id,
    "system_prompt": cfg.system_prompt,
    "llm_endpoint": cfg.llm_endpoint,
    "lakebase_project_id": None,  # cfg.lakebase_project_id,
}

git_sha = "abc"
run_id = "unset"

ts = ts = datetime.now().strftime("%Y-%m-%d")
with mlflow.start_run(
    run_name=f"valuation-agent-{ts}", tags={"git_sha": git_sha, "run_id": run_id}
) as run:
    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="../valuation_agent.py",
        resources=resources,
        input_example=test_request,
        model_config=model_config,
    )
    mlflow.log_metrics(results.metrics)

# COMMAND ----------
# register model to be ready for serving
model_name = f"{cfg.catalog}.{cfg.schema}.valuation_agent"

# databricks_model_serving: our agency has dependencies, so agent requires
# valuation_curator package which is a private library. databricks_model_serving takes
# context from serverless environment, that context is packaged and uploaded to a logged
# model and it's ready for staging.
registered_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=model_name,
    tags={"git_sha": git_sha, "run_id": run_id},
    env_pack="databricks_model_serving",
)
# here, to see the metrics, open the link and on the right side click on "model id" and
# "source run" (will also have system prompt from config file)
# NOTE: IN CASE OF FAILURE: when running multiple times in same session (not through jobs)
# this might fail because it keeps "concatenating" libraries to each other which leads to
# a large file that can break staging model for model serving (limitation 1gb model size).
# SOLUTION: on the config tab in db, reload environment

# COMMAND ----------

client = MlflowClient()
client.set_registered_model_alias(
    name=model_name,
    alias="latest-model",  # easier to retrieve the latest model later
    version=registered_model.version,
)
