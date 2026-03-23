# Databricks notebook source
# MAGIC %md
# MAGIC ## Lecture 2.2: PDF Parsing with AI Parse Documents
# MAGIC
# MAGIC ### Topics Covered:
# MAGIC - Downloading and storing PDFs
# MAGIC - AI Parse Documents for intelligent parsing
# MAGIC - Comparison with other PDF parsing tools
# MAGIC - Storing parsed content in Delta tables
# MAGIC - NOTE: THIS NOTEBOOK DOES NOT RUN LOCALLY: ai_parse_document requires db context

# COMMAND ----------
# %pip install ../valuation_curator-0.1.0-py3-none-any.whl
# COMMAND ----------

from valuation_curator.config import get_env, load_config
from valuation_curator.data_processor import DataProcessor
from databricks.connect import DatabricksSession
from loguru import logger

# COMMAND ----------

spark = DatabricksSession.builder.getOrCreate()
logger.info("✅ Using Databricks Connect Spark session")

env = get_env(spark)
cfg = load_config("../project_config.yml", env)

# Initialize the DataProcessor (reusable class from valuation_curator package)
processor = DataProcessor(spark=spark, config=cfg)

logger.info(f"Catalog: {cfg.catalog}, Schema: {cfg.schema}, Volume: {cfg.volume}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. PDF Parsing Tools Comparison
# MAGIC
# MAGIC | Tool | Pros | Cons | Best For |
# MAGIC |------|------|------|----------|
# MAGIC | **AI Parse Documents** | - AI-powered<br>- Handles complex layouts<br>- Integrated with Databricks<br>- Preserves structure | - Databricks-specific<br>- Cost per page | Complex documents, tables, multi-column |
# MAGIC | **PyPDF2** | - Simple<br>- Free<br>- Pure Python | - Poor with complex layouts<br>- No table extraction | Simple text extraction |
# MAGIC | **pdfplumber** | - Good table extraction<br>- Layout analysis | - Slower<br>- Manual tuning needed | Tables and structured data |
# MAGIC | **Apache Tika** | - Multi-format support<br>- Metadata extraction | - Java dependency<br>- Heavy | Multi-format processing |
# MAGIC | **Unstructured.io** | - ML-powered<br>- Good chunking | - External service<br>- API costs | Modern RAG pipelines |
# MAGIC
# MAGIC **AI Parse Documents** is the recommended choice for Databricks users due to its
# MAGIC integration and intelligent parsing capabilities.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Download PDFs, Parse, and Create Chunks
# MAGIC
# MAGIC The `DataProcessor` class from `valuation_curator.data_processor` handles the full
# MAGIC pipeline:
# MAGIC 1. Detect new/updated case folders from Google Drive using watermark
# MAGIC `max(processed)`
# MAGIC 2. Download case PDFs into `/Volumes/<catalog>/<schema>/<volume>/samples/case_xx/`
# MAGIC 3. Upsert metadata into `customs_valuation_metadata`
# MAGIC 4. Parse PDFs using AI Parse Documents (`ai_parse_document`) into
# MAGIC `ai_parsed_docs_table`
# MAGIC 5. Write chunks to `chunks_table` table with Change Data Feed enabled
# MAGIC
# MAGIC The same class is used in `resources/deployment_scripts/process_data.py`
# MAGIC for scheduled runs.

# COMMAND ----------

processor.process_and_save()
