import mlflow
from mlflow.models import ModelConfig

from valuation_curator.agent import ValuationAgent

# even though we already have the project config, we need to add this here because that's
# how mlflow works: we need this default values setting, and then when we load it we
# overwrite with the actual config values.
config = ModelConfig(
    development_config={
        "catalog": "customs",
        "schema": "customs_valuation",
        "genie_space_id": "01f12d0e51151a0f99caa978a12e068a",
        "system_prompt": "prompt placeholder",
        "llm_endpoint": "databricks-gpt-oss-120b",
        # "lakebase_project_id": "valuation-agent-lakebase",
    }
)

agent = ValuationAgent(
    llm_endpoint=config.get("llm_endpoint"),
    system_prompt=config.get("system_prompt"),
    catalog=config.get("catalog"),
    schema=config.get("schema"),
    genie_space_id=config.get("genie_space_id"),
    lakebase_project_id=config.get("lakebase_project_id"),
)
mlflow.models.set_model(agent)
