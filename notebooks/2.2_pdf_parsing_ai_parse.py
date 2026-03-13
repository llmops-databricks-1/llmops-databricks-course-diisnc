# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.2: PDF Parsing with AI Parse Documents
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Downloading and storing PDFs
# MAGIC - AI Parse Documents for intelligent parsing
# MAGIC - Comparison with other PDF parsing tools
# MAGIC - Storing parsed content in Delta tables

# COMMAND ----------
#%pip install ../arxiv_curator-0.1.0-py3-none-any.whl
# COMMAND ----------

from loguru import logger
from databricks.connect import DatabricksSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from arxiv_curator.config import load_config, get_env

# COMMAND ----------

spark = DatabricksSession.builder.getOrCreate()
logger.info("✅ Using Databricks Connect Spark session")

env = get_env(spark)
config_path = "../project_config.yml"

logger.info(f"Loading config from: {config_path}")
cfg = load_config(config_path, env)
catalog = cfg.catalog
schema = cfg.schema
volume = cfg.volume

# Create schema and volume if they don't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. PDF Parsing Tools Comparison
# MAGIC
# MAGIC | Tool | Pros | Cons | Best For |
# MAGIC |------|------|------|----------|
# MAGIC | **AI Parse Documents** | - AI-powered<br>- Handles complex layouts<br>- Integrated with Databricks<br>- Preserves structure | - Databricks-specific<br>- Cost per page | Complex documents, tables, multi-column |
# MAGIC | **PyPDF2** | - Simple<br>- Free<br>- Pure Python | - Poor with complex layouts<br>- No table extraction | Simple text extraction |
# MAGIC | **pdfplumber** | - Good table extraction<br>- Layout analysis | - Slower<br>- Manual tuning needed | Tables and structured data |
# MAGIC | **Apache Tika** | - Multi-format support<br>- Metadata extraction | - Java dependency<br>- Heavy | Multi-format processing |
# MAGIC | **Unstructured.io** | - ML-powered<br>- Good chunking | - External service<br>- API costs | Modern RAG pipelines |
# MAGIC
# MAGIC **AI Parse Documents** is the recommended choice for Databricks users due to its integration and intelligent parsing capabilities.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Downloading PDFs from arXiv
# MAGIC
# MAGIC We'll use arXiv as an example source of academic papers.

# COMMAND ----------
import os
import arxiv
import time

client = arxiv.Client()
metadata_table = f"{catalog}.{schema}.arxiv_papers"

# Read unprocessed papers from the table created in Lecture 1.3
papers_df = spark.table(metadata_table)
unprocessed_papers = papers_df.filter(F.col("processed").isNull())

logger.info(f"✅ Total papers in table: {papers_df.count()}")
logger.info(f"📄 Unprocessed papers: {unprocessed_papers.count()}")

# Limit to first 10 papers for demo purposes
papers_to_process = unprocessed_papers.limit(10).collect()
logger.info(f"📥 Will process {len(papers_to_process)} papers")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Download PDFs to Volume

# COMMAND ----------

# Create delta table with information about papers,
# including the location of the PDF file in volume storage

end = time.strftime("%Y%m%d%H%M", time.gmtime())
records = []
pdf_dir = f"/Volumes/{catalog}/{schema}/{volume}/{end}"
os.makedirs(pdf_dir, exist_ok=True)

logger.info(f"Downloading PDFs to: {pdf_dir}")

for paper_row in papers_to_process:
    paper_id = paper_row.arxiv_id
    
    try:
        # Get paper from arxiv using the arxiv_id
        search = arxiv.Search(id_list=[paper_id])
        paper = next(client.results(search))
        
        # Download PDF
        paper.download_pdf(dirpath=pdf_dir, filename=f"{paper_id}.pdf")
        
        # Collect metadata
        records.append({
            "paper_id": paper_id,
            "title": paper.title,
            "authors": [author.name for author in paper.authors],  # Array to match reference code
            "summary": paper.summary,
            "pdf_url": paper.pdf_url,
            "published": int(paper.published.strftime("%Y%m%d%H%M")),  # Long to match reference code
            "processed": int(f"{end}"),  # Long to match reference code
            "volume_path": f"{pdf_dir}/{paper_id}.pdf",
        })
        
        logger.info(f"✓ Downloaded: {paper_id} - {paper.title[:50]}...")
        
    except Exception as e:
        logger.warning(f"Paper {paper_id} was not successfully processed: {e}")
        pass
    
    time.sleep(3)  # Avoid rate limiting

logger.info(f"\n✅ Downloaded {len(records)} PDFs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Update arxiv_papers Table with PDF Paths

# COMMAND ----------

if len(records) > 0:
    # Create DataFrame with processed papers
    updates_schema = T.StructType([
        T.StructField("paper_id", T.StringType(), False),
        T.StructField("processed", T.LongType(), True),
        T.StructField("volume_path", T.StringType(), True),
    ])
    
    updates_data = [
        {"paper_id": r["paper_id"], "processed": r["processed"], "volume_path": r["volume_path"]}
        for r in records
    ]
    
    updates_df = spark.createDataFrame(updates_data, schema=updates_schema)
    updates_df.createOrReplaceTempView("pdf_updates")
    
    # Merge to update existing rows with volume_path and processed timestamp
    spark.sql(f"""
        MERGE INTO {metadata_table} target
        USING pdf_updates source
        ON target.arxiv_id = source.paper_id
        WHEN MATCHED THEN UPDATE SET
            target.processed = source.processed,
            target.volume_path = source.volume_path
    """)
    
    logger.info(f"✓ Updated {len(records)} records in {metadata_table} with PDF paths")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. AI Parse Documents
# MAGIC
# MAGIC Now we'll use Databricks' AI Parse Documents feature to intelligently parse the PDFs.

# COMMAND ----------

if len(records) > 0:
    # Create table for parsed documents
    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {catalog}.{schema}.ai_parsed_docs (
        path STRING,
        parsed_content STRING,
        processed LONG
    )
    """)
    
    # Use AI Parse Documents to parse PDFs
    spark.sql(f"""
    INSERT INTO {catalog}.{schema}.ai_parsed_docs
    SELECT
        path,
        ai_parse_document(content) AS parsed_content,
        {end} AS processed
    FROM READ_FILES(
        "{pdf_dir}",
        format => 'binaryFile'
    )
    """)
    
    logger.info(f"✓ PDFs parsed and saved to {catalog}.{schema}.ai_parsed_docs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Extract and Clean Chunks from Parsed Content

# COMMAND ----------

import re
import json
from pyspark.sql.types import ArrayType, StringType, StructField, StructType
from pyspark.sql.functions import col, concat_ws, explode, udf

# UDF to extract chunks from parsed_content JSON
def extract_chunks(parsed_content_json: str) -> list:
    parsed_dict = json.loads(parsed_content_json)
    chunks = []
    
    for element in parsed_dict.get("document", {}).get("elements", []):
        if element.get("type") == "text":
            chunk_id = element.get("id", "")
            content = element.get("content", "")
            chunks.append((chunk_id, content))
    return chunks

chunk_schema = ArrayType(
    StructType([
        StructField("chunk_id", StringType(), True),
        StructField("content", StringType(), True),
    ])
)
extract_chunks_udf = udf(extract_chunks, chunk_schema)

# UDF to extract paper_id from path
def extract_paper_id(path):
    return path.replace(".pdf", "").split("/")[-1]

extract_paper_id_udf = udf(extract_paper_id, StringType())

# UDF to clean chunk text
def clean_chunk(text: str) -> str:
    # Fix hyphenation across line breaks: "docu-\nments" => "documents"
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    
    # Collapse internal newlines into spaces
    t = re.sub(r"\s*\n\s*", " ", t)
    
    # Collapse repeated whitespace
    t = re.sub(r"\s+", " ", t)
    
    return t.strip()

clean_chunk_udf = udf(clean_chunk, StringType())

# COMMAND ----------

# Load parsed documents
df = spark.table(f"{catalog}.{schema}.ai_parsed_docs").where(f"processed = {end}")

# Load metadata table
metadata_df = spark.table(metadata_table).select(
    col("arxiv_id").alias("paper_id"),  # Rename arxiv_id to paper_id for join
    col("title"),
    col("summary"),
    concat_ws(", ", col("authors")).alias("authors"),  # Join array to string
    (col("published") / 100000000).cast("int").alias("year"),
    ((col("published") % 100000000) / 1000000).cast("int").alias("month"),
    ((col("published") % 1000000) / 10000).cast("int").alias("day"),
)

# Create the transformed table
chunks_df = (
    df.withColumn("paper_id", extract_paper_id_udf(col("path")))
    .withColumn("chunks", extract_chunks_udf(col("parsed_content")))
    .withColumn("chunk", explode(col("chunks")))
    .select(
        col("paper_id"),
        col("chunk.chunk_id").alias("chunk_id"),
        clean_chunk_udf(col("chunk.content")).alias("text"),
        concat_ws("_", col("paper_id"), col("chunk.chunk_id")).alias("id"),
    )
    .join(metadata_df, "paper_id", "left")
)

logger.info(f"✓ Extracted and cleaned chunks")
chunks_df.show(5, truncate=50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Write Chunks to Table

# COMMAND ----------

# Write to table
chunks_table = f"{catalog}.{schema}.arxiv_chunks"
chunks_df.write.mode("append").saveAsTable(chunks_table)

logger.info(f"✓ Chunks saved to {chunks_table}")

# COMMAND ----------

# Enable Change Data Feed for incremental processing
spark.sql(f"""
ALTER TABLE {chunks_table}
SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

logger.info(f"✓ Change Data Feed enabled on {chunks_table}")
