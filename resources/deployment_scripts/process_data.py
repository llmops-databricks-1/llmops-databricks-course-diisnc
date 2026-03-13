# Databricks notebook source
# MAGIC %md
# MAGIC # Data Processing Pipeline
# MAGIC
# MAGIC This notebook processes arXiv papers and syncs the vector search index.
# MAGIC Runs on a schedule to keep the knowledge base up to date.

# COMMAND ----------

import yaml
from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.config import load_config, get_env
from arxiv_curator.vector_search import VectorSearchManager
from arxiv_curator.utils.common import get_widget

# Get environment from widget (set by workflow)
env = get_widget("env", "dev")
xs
# Load configuration
cfg = load_config("../../project_config.yml", env=env)
logger.info("Configuration loaded:")
logger.info(f"  Environment: {env}")
logger.info(f"  Catalog: {cfg.catalog}")
logger.info(f"  Schema: {cfg.schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process Data
# MAGIC
# MAGIC In a real implementation, this would:
# MAGIC 1. Download new arXiv papers
# MAGIC 2. Parse PDFs with AI
# MAGIC 3. Chunk documents
# MAGIC 4. Generate embeddings
# MAGIC
# MAGIC For the course, we'll just sync the vector search index.

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Sync vector search index
logger.info("Syncing vector search index...")
vector_search_manager = VectorSearchManager(
    catalog=cfg.catalog,
    schema=cfg.schema,
    endpoint_name=cfg.vector_search_endpoint,
    index_name=cfg.vector_search_index,
)

# Trigger index sync
vector_search_manager.sync_index()

logger.info("✓ Data processing complete!")
