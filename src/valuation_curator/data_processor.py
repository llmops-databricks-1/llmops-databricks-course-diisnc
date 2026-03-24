"""
Google Drive folder
    ↓ (download_and_store_papers)
PDFs in Volume + customs_valuation_metadata table
   ↓ (parse_pdfs_with_ai)
ai_parsed_docs_table (JSON)
   ↓ (process_chunks)
chunks_table (clean text + metadata)
   ↓ (VectorSearchManager - separate class) (2.4 notebook)
Vector Search Index (embeddings)
"""

import json
import datetime
import os
import re
import shutil
import time
from urllib import parse, request

from valuation_curator.config import ProjectConfig
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql.functions import (
    col,
    concat_ws,
    explode,
    udf,
)
from pyspark.sql.types import ArrayType, StringType, StructField, StructType


class DataProcessor:
    """
    DataProcessor handles the complete incremental pipeline for customs valuation cases.

    Responsibilities:
        - Incremental discovery of new/updated case folders in Google Drive
        - Downloading case PDFs into Databricks Volume
        - Upserting case metadata into `customs_valuation_metadata`
        - Parsing PDFs using `ai_parse_document`
        - Extracting, cleaning, and saving text chunks to `chunks_table`

    Incremental Logic:
        Each run is identified by a `run_processed` timestamp (format: YYYYMMDDHHMM).
        `_get_range_start()` reads `max(processed)` from the metadata table to determine
        where the last run ended. If no watermark exists, it falls back to 5 days ago.
        Only Drive folders modified after the watermark are fetched.

    Metadata Merge Semantics:
        The metadata table is keyed by `id` (case folder name, e.g. case_00).
        - Existing rows: updated (processed, volume_path)
        - New rows: inserted
        Fields `processed` and `volume_path` are set here and must not be overwritten
        by the bootstrap notebook (1.3).

    Parse Scope:
        `parse_pdfs_with_ai()` only processes cases where `processed == run_processed`,
        i.e. cases downloaded in the current run.

    Output Tables:
        - `customs_valuation_metadata`: case-level metadata (upsert)
        - `ai_parsed_docs_table`: raw AI-parsed output per PDF (append)
        - `chunks_table`: cleaned text chunks joined with metadata (append, CDF enabled)

    Recovery / Backfill:
        Use notebook `1.3_valuation_data_ingestion` to register Volume files that
        were added manually outside this pipeline. Then run this class to parse and
        chunk any newly registered cases.

    Note:
        `ai_parse_document` requires a Databricks runtime context and will not run
        locally. Secrets for Google Drive must be configured under scope='gdrive'.
    """

    def __init__(self, spark: SparkSession, config: ProjectConfig) -> None:
        """
        Initialize DataProcessor with Spark session and configuration.

        Args:
            spark: SparkSession instance
            config: ProjectConfig object with table configurations
        """
        self.spark = spark
        self.cfg = config
        self.catalog = config.catalog
        self.schema = config.schema
        self.volume = config.volume

        # key to track each run
        self.end = time.strftime("%Y%m%d%H%M", time.gmtime())
        self.run_processed = int(self.end)

        # fixed samples root where case folders are stored (case_00, case_01, ...)
        self.pdf_dir = f"/Volumes/{self.catalog}/{self.schema}/{self.volume}/samples"
        os.makedirs(self.pdf_dir, exist_ok=True)  # only created if not exists

        # where metadata is stored
        self.docs_table = f"{self.catalog}.{self.schema}.customs_valuation_metadata"
        # where result of ai_parse_document will be stored
        self.parsed_table = f"{self.catalog}.{self.schema}.ai_parsed_docs_table"

    @staticmethod
    def _to_drive_timestamp(value: datetime.datetime) -> str:
        """Convert datetime to Google Drive RFC3339 UTC format."""
        return value.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _drive_get_json(url: str, params: dict[str, str]) -> dict:
        """Execute GET request against Google Drive API and return JSON payload."""
        query = parse.urlencode(params)
        with request.urlopen(f"{url}?{query}", timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _download_drive_file(file_id: str, destination: str, api_key: str) -> None:
        """Download a single file from Google Drive to destination path."""
        file_url = (
            f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"
        )
        tmp_path = f"{destination}.tmp"
        with request.urlopen(file_url, timeout=120) as response:
            with open(tmp_path, "wb") as output_file:
                shutil.copyfileobj(response, output_file)
        os.replace(tmp_path, destination)

    def _get_drive_credentials(self) -> tuple[str, str]:
        """Load Google Drive API key and root folder ID from Databricks secrets."""
        try:
            from pyspark.dbutils import DBUtils

            dbutils = DBUtils(self.spark)
            api_key = dbutils.secrets.get(scope="gdrive", key="api-key")
            folder_id = dbutils.secrets.get(scope="gdrive", key="folder-id")
            return api_key, folder_id
        except Exception:
            api_key = os.getenv("GOOGLE_DRIVE_API_KEY")
            folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
            if api_key and folder_id:
                return api_key, folder_id
            raise ValueError(
                "Missing Google Drive credentials. Configure Databricks secrets "
                "scope='gdrive' keys 'api-key' and 'folder-id', or set env vars "
                "GOOGLE_DRIVE_API_KEY and GOOGLE_DRIVE_FOLDER_ID."
            )

    def _list_case_folders(
        self,
        root_folder_id: str,
        api_key: str,
        start: datetime.datetime,
        end: datetime.datetime,
    ) -> list[dict]:
        """List Drive case folders modified in the given interval."""
        folders: list[dict] = []
        page_token: str | None = None

        start_str = self._to_drive_timestamp(start)
        end_str = self._to_drive_timestamp(end)

        while True:
            query = (
                f"'{root_folder_id}' in parents"
                " and mimeType='application/vnd.google-apps.folder'"
                " and trashed=false"
                f" and modifiedTime > '{start_str}'"
                f" and modifiedTime <= '{end_str}'"
            )
            params = {
                "q": query,
                "fields": "nextPageToken,files(id,name,modifiedTime)",
                "pageSize": "1000",
                "key": api_key,
            }
            if page_token is not None:
                params["pageToken"] = page_token

            payload = self._drive_get_json(
                "https://www.googleapis.com/drive/v3/files", params
            )
            for folder in payload.get("files", []):
                if folder["name"].startswith("case_"):
                    folders.append(folder)

            page_token = payload.get("nextPageToken")
            if page_token is None:
                break

        return folders

    def _list_case_pdfs(self, case_folder_id: str, api_key: str) -> list[dict]:
        """List PDF files inside a case folder."""
        files: list[dict] = []
        page_token: str | None = None

        while True:
            query = (
                f"'{case_folder_id}' in parents"
                " and mimeType='application/pdf'"
                " and trashed=false"
            )
            params = {
                "q": query,
                "fields": "nextPageToken,files(id,name)",
                "pageSize": "1000",
                "key": api_key,
            }
            if page_token is not None:
                params["pageToken"] = page_token

            payload = self._drive_get_json(
                "https://www.googleapis.com/drive/v3/files", params
            )
            files.extend(payload.get("files", []))

            page_token = payload.get("nextPageToken")
            if page_token is None:
                break

        return files

    def _get_range_start(self) -> datetime.datetime:
        """
        Get start timestamp for incremental fetch.

        If customs_valuation_metadata exists, use max(processed) as start.
        Otherwise, use 5 days ago.

        Returns:
            UTC datetime for start of sync interval.
        """

        if self.spark.catalog.tableExists(self.docs_table):
            result = self.spark.sql(
                f"""
                SELECT max(processed)
                FROM {self.docs_table}
            """
            ).collect()
            max_processed = result[0][0]
            if max_processed is not None:
                start = datetime.datetime.strptime(
                    str(max_processed), "%Y%m%d%H%M"
                ).replace(tzinfo=datetime.timezone.utc)
                logger.info(
                    "Found existing customs_valuation_metadata table. "
                    f"Starting from: {start.isoformat()}"
                )
                return start

        start = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=5)
        logger.info(
            "No existing customs_valuation_metadata table. "
            f"Starting from 5 days ago: {start.isoformat()}"
        )
        return start

    def download_and_store_papers(self) -> list[dict] | None:
        """
        Download new/updated case PDFs from Google Drive and store metadata
        in customs_valuation_metadata table.

        Returns:
            List of case metadata dictionaries if cases were downloaded,
            otherwise None.
        """
        start = self._get_range_start()
        end = datetime.datetime.now(datetime.timezone.utc)
        api_key, root_folder_id = self._get_drive_credentials()

        case_folders = self._list_case_folders(
            root_folder_id=root_folder_id,
            api_key=api_key,
            start=start,
            end=end,
        )

        if len(case_folders) == 0:
            logger.info(
                "No new case folders found in Google Drive for interval "
                f"{start.isoformat()} to {end.isoformat()}."
            )
            return None

        # Download files and build metadata rows
        records = []

        for folder in case_folders:
            case_id = folder["name"]
            case_path = os.path.join(self.pdf_dir, case_id)
            os.makedirs(case_path, exist_ok=True)

            invoice_path = None
            declaration_path = None
            royalty_path = None

            for pdf_file in self._list_case_pdfs(folder["id"], api_key):
                filename = pdf_file["name"]
                local_path = os.path.join(case_path, filename)

                try:
                    self._download_drive_file(pdf_file["id"], local_path, api_key)
                except Exception as error:
                    logger.warning(
                        f"Failed to download {filename} for {case_id}: {error}"
                    )
                    continue

                if filename.startswith("invoice_"):
                    invoice_path = local_path
                elif filename.startswith("declaration_"):
                    declaration_path = local_path
                elif filename.startswith("royalty_"):
                    royalty_path = local_path

            if invoice_path and declaration_path:
                records.append(
                    {
                        "id": case_id,
                        "invoice_path": invoice_path,
                        "declaration_path": declaration_path,
                        "royalty_path": royalty_path,
                        "ingestion_timestamp": datetime.datetime.now(
                            datetime.timezone.utc
                        )
                        .replace(tzinfo=None)
                        .isoformat(),
                        "processed": self.run_processed,
                        "volume_path": case_path,
                    }
                )
            else:
                logger.warning(
                    f"Skipping case {case_id}: missing invoice or declaration PDF."
                )

        # Only process if we have records
        if len(records) == 0:
            logger.info("No complete new cases found after download.")
            return None

        logger.info(f"Downloaded {len(records)} cases to {self.pdf_dir}")

        # Create DataFrame with metadata schema aligned to notebook 1.3
        schema = T.StructType(
            [
                T.StructField("id", T.StringType(), False),
                T.StructField("invoice_path", T.StringType(), True),
                T.StructField("declaration_path", T.StringType(), True),
                T.StructField("royalty_path", T.StringType(), True),
                T.StructField("ingestion_timestamp", T.StringType(), True),
                T.StructField("processed", T.LongType(), True),
                T.StructField("volume_path", T.StringType(), True),
            ]
        )

        metadata_df = self.spark.createDataFrame(records, schema=schema)

        # Create table if it doesn't exist, then upsert by case id
        metadata_df.write.format("delta").mode("ignore").saveAsTable(self.docs_table)

        metadata_df.createOrReplaceTempView("new_cases")
        self.spark.sql(
            f"""
            MERGE INTO {self.docs_table} target
            USING new_cases source
            ON target.id = source.id
            WHEN MATCHED THEN UPDATE SET
                target.processed = source.processed,
                target.volume_path = source.volume_path
            WHEN NOT MATCHED THEN INSERT (
                id,
                invoice_path,
                declaration_path,
                royalty_path,
                ingestion_timestamp,
                processed,
                volume_path
            ) VALUES (
                source.id,
                source.invoice_path,
                source.declaration_path,
                source.royalty_path,
                source.ingestion_timestamp,
                source.processed,
                source.volume_path
            )
        """
        )
        logger.info(f"Merged {len(records)} case records into {self.docs_table}")
        return records

    def parse_pdfs_with_ai(self) -> None:
        """
        Parse PDFs using ai_parse_document and store in ai_parsed_docs table.

        """

        self.spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {self.parsed_table} (
                path STRING,
                parsed_content STRING,
                processed LONG
            )
        """
        )

        current_case_paths = self.spark.sql(
            f"""
            SELECT volume_path
            FROM {self.docs_table}
            WHERE processed = {self.run_processed}
        """
        ).collect()

        if len(current_case_paths) == 0:
            logger.info("No new case paths to parse for this run.")
            return
        for row in current_case_paths:
            case_path = row["volume_path"]
            self.spark.sql(
                f"""
                INSERT INTO {self.parsed_table}
                SELECT
                    path,
                    ai_parse_document(content) AS parsed_content,
                    {self.run_processed} AS processed
                FROM READ_FILES(
                    "{case_path}",
                    format => 'binaryFile'
                )
            """
            )

        logger.info(
            f"Parsed PDFs for {len(current_case_paths)} cases into {self.parsed_table}"
        )

    @staticmethod
    def _extract_chunks(parsed_content_json: str) -> list[tuple[str, str]]:
        """
        Extract chunks from parsed_content JSON.

        Args:
            parsed_content_json: JSON string containing
            parsed document structure

        Returns:
            List of tuples containing (chunk_id, content)
        """
        parsed_dict = json.loads(parsed_content_json)
        chunks = []

        # Extract text, table, section_header, and other content-bearing elements
        for element in parsed_dict.get("document", {}).get("elements", []):
            element_type = element.get("type")
            if element_type in ("text", "table", "section_header"):
                chunk_id = element.get("id", "")
                content = element.get("content", "")
                if content:  # Only include if content is not empty
                    chunks.append((chunk_id, content))

        return chunks

    @staticmethod
    def _extract_cases_id(path: str) -> str:
        """
        Extract case ID from file path.

        Args:
            path: File path containing case directory (e.g. "/.../case_00/invoice_00.pdf")

        Returns:
            Case ID extracted from the path
        """
        parts = path.replace("\\", "/").split("/")
        for part in reversed(parts):
            if part.startswith("case_"):
                return part
        return path.replace(".pdf", "").split("/")[-1]

    @staticmethod
    def _extract_document_id(path: str) -> str:
        """
        Extract document ID from file path.

        Args:
            path: File path containing PDF filename

        Returns:
            Document ID (filename without .pdf extension)
        """
        return os.path.basename(path).replace(".pdf", "")

    @staticmethod
    def _clean_chunk(text: str) -> str:
        """
        Clean and normalize chunk text
        Args:
            text: Raw text content

        Returns:
            Cleaned text content
        """
        # Fix hyphenation across line breaks:
        # "docu-\nments" => "documents"
        t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

        # Collapse internal newlines into spaces
        t = re.sub(r"\s*\n\s*", " ", t)

        # Collapse repeated whitespace
        t = re.sub(r"\s+", " ", t)

        return t.strip()

    def process_chunks(self) -> None:
        """
        Process parsed documents to extract and clean chunks.
        Reads from ai_parsed_docs table and saves to chunks table.
        """
        logger.info(
            f"Processing parsed documents from "
            f"{self.parsed_table} for run {self.run_processed}"
        )

        df = self.spark.table(self.parsed_table).where(
            f"processed = {self.run_processed}"
        )

        # Define schema for the extracted chunks
        chunk_schema = ArrayType(
            StructType(
                [
                    StructField("chunk_id", StringType(), True),
                    StructField("content", StringType(), True),
                ]
            )
        )

        extract_chunks_udf = udf(self._extract_chunks, chunk_schema)
        extract_cases_id_udf = udf(self._extract_cases_id, StringType())
        extract_document_id_udf = udf(self._extract_document_id, StringType())
        clean_chunk_udf = udf(self._clean_chunk, StringType())

        metadata_df = self.spark.table(self.docs_table).select(
            col("id").alias("case_id"),
            col("invoice_path"),
            col("declaration_path"),
            col("royalty_path"),
        )

        # Create the transformed dataframe
        chunks_df = (
            df.withColumn("case_id", extract_cases_id_udf(col("path")))
            .withColumn("document_id", extract_document_id_udf(col("path")))
            .withColumn("chunks", extract_chunks_udf(col("parsed_content")))
            .withColumn("chunk", explode(col("chunks")))
            .select(
                col("case_id"),
                col("document_id"),
                col("path").alias("document_path"),
                col("chunk.chunk_id").alias("chunk_id"),
                clean_chunk_udf(col("chunk.content")).alias("text"),
                concat_ws(
                    "_",
                    col("case_id"),
                    col("document_id"),
                    col("chunk.chunk_id"),
                ).alias("id"),
            )
            .join(metadata_df, "case_id", "left")
            .withColumn(
                "metadata",
                concat_ws(
                    " | ",
                    col("invoice_path"),
                    col("declaration_path"),
                    col("royalty_path"),
                ),
            )
        )

        # Write to table
        chunks_table = f"{self.catalog}.{self.schema}.chunks_table"
        chunks_df.write.mode("append").saveAsTable(chunks_table)
        logger.info(f"Saved chunks to {chunks_table}")

        # Enable Change Data Feed: important since vector search index will be updated
        # based on changes in the chunks table. Allows to update only based on new content
        # that's being added instead of reprocessing everything.
        self.spark.sql(
            f"""
            ALTER TABLE {chunks_table}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """
        )
        logger.info(f"Change Data Feed enabled for {chunks_table}")

    def process_and_save(self) -> None:
        """
        Complete workflow: download case PDFs, parse, and process chunks.
        """
        # Step 1: Download case files and store metadata
        records = self.download_and_store_papers()

        # Only continue if we have new cases
        if records is None:
            logger.info("No new cases to process. Exiting.")
            return

        # Step 2: Parse PDFs with ai_parse_document
        # Function ai_parse_document already does chunking but it needs to be processed to
        # extract them (done in next step)
        self.parse_pdfs_with_ai()
        logger.info("Parsed documents.")

        # Step 3: Process chunks
        self.process_chunks()
        logger.info("Processing complete!")
