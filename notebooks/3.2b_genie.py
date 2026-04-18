# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.2b: Genie Space Integration
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Creating a SQL warehouse for Genie
# MAGIC - Configuring a Genie space with data sources
# MAGIC - Starting conversations with Genie
# MAGIC - Using Genie for natural language queries
# MAGIC
# MAGIC **What is Genie?**
# MAGIC - Databricks Genie is an AI-powered data analyst
# MAGIC - Converts natural language questions to SQL queries
# MAGIC - Executes queries and returns results
# MAGIC - Can be integrated with agents via MCP
# MAGIC Note: in this nb, Genie was connected to metadata and not vector search table.
# MAGIC Genie should be connected to structured data to generate SQL, not embeddings
# MAGIC or vector indexes.

# COMMAND ----------
import json

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql
from databricks.sdk.service.sql import CreateWarehouseRequestWarehouseType
from loguru import logger
from pyspark.sql import SparkSession

from valuation_curator.config import get_env, load_config

spark = SparkSession.builder.getOrCreate()

# Load configuration
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

catalog = cfg.catalog
schema = cfg.schema


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Check for Existing Genie Space
# MAGIC
# MAGIC First, check if we already have a Genie space configured.

# COMMAND ----------


w = WorkspaceClient()

# Check if genie_space_id is configured
if hasattr(cfg, "genie_space_id") and cfg.genie_space_id:
    logger.info(f"Using existing Genie Space from config: {cfg.genie_space_id}")
    space_id = cfg.genie_space_id
    USE_EXISTING_SPACE = True
else:
    logger.info("No Genie Space configured, will create a new one")
    USE_EXISTING_SPACE = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create SQL Warehouse (if needed)
# MAGIC
# MAGIC Genie requires a SQL warehouse to execute queries.
# MAGIC Skip this if using an existing space.
# MAGIC Note: this config is for serverless.

# COMMAND ----------

if not USE_EXISTING_SPACE:
    # Create a new warehouse for the Genie space (NEEDS UPDATE)
    created = w.warehouses.create(
        name="__2XS_valuation_warehouse",
        cluster_size="2X-Small",
        max_num_clusters=1,
        auto_stop_mins=10,
        warehouse_type=CreateWarehouseRequestWarehouseType("PRO"),
        enable_serverless_compute=True,
        tags=sql.EndpointTags(
            custom_tags=[sql.EndpointTagPair(key="Project", value="valuation_curator")]
        ),
    ).result()
    warehouse_id = created.id
    logger.info(f"Created warehouse: {warehouse_id}")
else:
    # Use warehouse from config
    warehouse_id = cfg.warehouse_id
    logger.info(f"Using existing warehouse: {warehouse_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Configure Genie Space
# MAGIC
# MAGIC Define which tables and columns Genie can access.
# MAGIC Skip this if using an existing space.

# COMMAND ----------

# Configure the Genie space with arxiv_papers table ------ (NEEDS UPDATE) ------
# Genie should be connected to structured data to generate SQL, not embeddings or
# vector indexes.
serialized_space = {
    "version": 1,
    "data_sources": {
        "tables": [
            {
                "identifier": f"{catalog}.{schema}.arxiv_papers",
                "column_configs": [
                    {"column_name": "authors"},
                    {"column_name": "ingest_ts", "get_example_values": True},
                    {"column_name": "paper_id", "get_example_values": True},
                    {
                        "column_name": "pdf_url",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                    {"column_name": "processed", "get_example_values": True},
                    {"column_name": "published", "get_example_values": True},
                    {
                        "column_name": "summary",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                    {
                        "column_name": "title",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                    {
                        "column_name": "volume_path",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                ],
            }
        ]
    },
}

if not USE_EXISTING_SPACE:
    space = w.genie.create_space(
        warehouse_id=warehouse_id,
        serialized_space=json.dumps(serialized_space),
        title="arxiv-curator-space",
    )
    space_id = space.space_id
    logger.info(f"Created new Genie Space: {space_id}")
else:
    logger.info(f"Using existing Genie Space: {space_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Verify Genie Space

# COMMAND ----------

space = w.genie.get_space(space_id=space_id, include_serialized_space=True)
logger.info(f"Genie Space ID: {space_id}")
logger.info(f"Space config: {json.loads(space.serialized_space)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Start a Conversation
# MAGIC
# MAGIC Ask Genie a natural language question about the data.
# MAGIC The questions can be accessed in the monitoring tab in the Genie UI.

# COMMAND ----------

conversation = w.genie.start_conversation_and_wait(
    space_id=space.space_id, content="What was the date of the last ingested files?"
)

conversation.as_dict()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Continue the Conversation
# MAGIC
# MAGIC Ask follow-up questions in the same conversation.

# COMMAND ----------

message = w.genie.create_message_and_wait(
    space_id=space.space_id,
    conversation_id=conversation.conversation_id,
    content="For the files ingested in that date, list their ids.",
)

message.as_dict()
