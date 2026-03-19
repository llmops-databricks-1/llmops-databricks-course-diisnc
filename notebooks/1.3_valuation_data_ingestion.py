# Databricks notebook source
import os
from datetime import datetime

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, StringType, StructField, StructType

from valuation_curator.config import get_env, load_config

# COMMAND ----------
# Create Spark session
spark = SparkSession.builder.getOrCreate()

# Load config
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

CATALOG = cfg.catalog
SCHEMA = cfg.schema
VOLUME = cfg.volume
TABLE_NAME = "customs_valuation_metadata"

# Create a volume for storing PDFs - one time setup
# Step 1: Create the catalog
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
logger.info(f"Catalog '{CATALOG}' ready")

# Step 2: Create the schema within the catalog
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
logger.info(f"Schema '{CATALOG}.{SCHEMA}' ready")

# Step 3: Create the volume
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
logger.info(f"Volume {CATALOG}.{SCHEMA}.{VOLUME} ready")


# COMMAND ----------
# Fetch valuation metadata from the volume.
# The costums valuation is done by analysing 3 files:
# - the invoice of the goods shipped
# - the royalty contracts (if applicable)
# - the declaration where all costs should be declared: total of goods,
#   royalties (if applicable), and any taxes
# NOTE: will only run in databricks
def fetch_valuation_metadata(volume_path: str) -> list[dict]:
    """
    Fetch valuation metadata from PDFs stored in Databricks Volume.

    Args:
        volume_path: Path to the samples folder in the Volume

    Returns:
        List of valuation metadata dictionaries
    """
    cases = {}

    if not os.path.exists(volume_path):
        logger.warning(f"Volume path not found: {volume_path}")
        return []

    # Walk through all subdirectories (case_00, case_01, ...)
    for dirpath, _, filenames in os.walk(volume_path):
        case_id = os.path.basename(dirpath)

        if not case_id.startswith("case_"):
            continue

        if case_id not in cases:
            cases[case_id] = {
                "id": case_id,
                "invoice_path": None,
                "declaration_path": None,
                "royalty_path": None,
                "ingestion_timestamp": datetime.now().isoformat(),
                "processed": None,  # Will be set in Lecture 2.2?
                "volume_path": None,  # Will be set in Lecture 2.2?
            }

        # Map PDF files to their respective paths
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                filepath = os.path.join(dirpath, filename)

                if filename.startswith("invoice_"):
                    cases[case_id]["invoice_path"] = filepath
                elif filename.startswith("declaration_"):
                    cases[case_id]["declaration_path"] = filepath
                elif filename.startswith("royalty_"):
                    cases[case_id]["royalty_path"] = filepath

    # Only return cases that have both invoice and declaration
    complete_cases = [
        case
        for case in cases.values()
        if case["invoice_path"] and case["declaration_path"]
    ]

    return complete_cases


logger.info("Fetching case metadata from Volume...")
path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/samples"
cases = fetch_valuation_metadata(path)
logger.info(f"Found {len(cases)} complete cases")
if cases:
    logger.info("Sample case:")
    logger.info(f"ID: {cases[0]['id']}")
    logger.info(f"Invoice: {cases[0]['invoice_path']}")
    logger.info(f"Declaration: {cases[0]['declaration_path']}")
    logger.info(f"Royalty: {cases[0]['royalty_path']}")

# cases

# COMMAND ----------
# Create Delta Table in Unity Catalog
# Store the case metadata in a Delta table for downstream processing.

# Define schema
schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("invoice_path", StringType(), True),
        StructField("declaration_path", StringType(), True),
        StructField("royalty_path", StringType(), True),
        StructField("ingestion_timestamp", StringType(), True),
        StructField("processed", LongType(), True),  # Will be set in Lecture 2.2?
        StructField("volume_path", StringType(), True),  # Will be set in Lecture 2.2?
    ]
)

# Create DataFrame
df = spark.createDataFrame(cases, schema=schema)

# Write to Delta table
table_path = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(
    table_path
)

logger.info(f"Created Delta table: {table_path}")
logger.info(f"Records: {df.count()}")

# display(df.limit(5))

# COMMAND ----------
# Verify the Data

# Read back the table
cases_df = spark.table(f"{CATALOG}.{SCHEMA}.{TABLE_NAME}")

logger.info(f"Table: {CATALOG}.{SCHEMA}.{TABLE_NAME}")
logger.info(f"Total cases: {cases_df.count()}")
logger.info("Schema:")
cases_df.printSchema()

logger.info("Sample records:")
cases_df.select("id", "invoice_path", "declaration_path", "royalty_path").show(
    5, truncate=50
)

# COMMAND ----------
# Data Statistics

logger.info("Cases by completeness:")
cases_df.select(
    cases_df.id,
    cases_df.invoice_path.isNotNull().alias("has_invoice"),
    cases_df.declaration_path.isNotNull().alias("has_declaration"),
    cases_df.royalty_path.isNotNull().alias("has_royalty"),
).groupBy("has_invoice", "has_declaration", "has_royalty").count().orderBy(
    "count", ascending=False
).show()
