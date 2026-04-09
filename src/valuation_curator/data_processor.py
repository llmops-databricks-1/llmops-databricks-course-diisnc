"""
NOTE: this class is too big and could be splitted in multiple ones according to logic,
but I'll keep it to match the structure of the course.
Google Drive folder
    ↓ (download_and_store_papers)
PDFs in Volume + customs_valuation_metadata table
    ↓ (parse_pdfs_with_ai)
ai_parsed_docs_table (JSON)
    ↓ (translate_parsed_docs)
    ↓ (process_chunks + metadata enrichment)
chunks_table (clean text + metadata)
    ↓ (VectorSearchManager - separate class) (2.4 notebook)
Vector Search Index (embeddings)
"""

import datetime
import html
import json
import os
import re
import shutil
import time
from urllib import parse, request

import mlflow.deployments
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql.functions import col, concat_ws, explode, first, udf
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from valuation_curator.config import ProjectConfig


class DataProcessor:
    """
    DataProcessor handles the complete incremental pipeline for customs valuation cases.

    Responsibilities:
        - Incremental discovery of new/updated case folders in Google Drive
        - Downloading case PDFs into Databricks Volume
        - Upserting case metadata into `customs_valuation_metadata`
        - Parsing PDFs using `ai_parse_document`
        - Detecting source document language and translating parsed content to English
        - Extracting pdf metadata from parsed content
        - Extracting and cleaning English chunks to `chunks_table`

    Incremental Logic:
        Each run is identified by a `run_processed` timestamp (format: YYYYMMDDHHMM).
        `_get_range_start()` reads `max(processed)` from the metadata table to determine
        where the last run ended. If no watermark exists, it falls back to 5 days ago.
        Only Drive folders modified after the watermark are fetched.

    Metadata Merge Semantics:
        The metadata table key is `id` (case folder name, e.g. case_00).
        - Existing rows: updated (processed, volume_path)
        - New rows: inserted

    Parse Scope:
        `parse_pdfs_with_ai()` only processes cases where `processed == run_processed`,
        i.e. cases downloaded in the current run.

    Translation Scope:
        `translate_parsed_docs()` processes parsed rows for `processed == run_processed`
        and writes:
        - `parsed_content_translated` (translated JSON structure)
        - `source_language` (ISO 639-1 code, e.g. en/es/de)
        - `translation_applied` (True when source language is not English)
        Translation strategy is document-level, before chunking:
        - 1 LLM call per document for language detection + translation
        - Could've used cheap python detection method first and reduce LLM calls to
        non-English docs, but it was not working decently.

    Output Tables:
        - `customs_valuation_metadata`: case-level metadata (upsert)
        - `ai_parsed_docs_table`: raw AI-parsed output per PDF (append) + translated
           content + language metadata
        - `chunks_table`: cleaned English chunks + language metadata + business metadata
           (append, CDF enabled)

    Recovery / Backfill:
        Use notebook `1.3_valuation_data_ingestion` to register Volume files that
        were added manually outside this pipeline. Then run this class to parse and
        chunk any newly registered cases.

    Note:
        - `ai_parse_document` will not run locally.
        - Secrets for Google Drive must be configured under scope='gdrive'.
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

    # ----------------------------- Google Drive Interaction -----------------------------
    @staticmethod
    def _to_drive_timestamp(value: datetime.datetime) -> str:
        """Convert datetime to Google Drive RFC3339 UTC format."""
        return value.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _drive_get_json(url: str, params: dict[str, str]) -> dict:
        """Execute GET request against Google Drive API and return JSON payload."""
        query = parse.urlencode(params)
        with request.urlopen(f"{url}?{query}", timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _download_drive_file(file_id: str, destination: str, api_key: str) -> None:
        """Download a single file from Google Drive to destination path."""
        file_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"
        tmp_path = f"{destination}.tmp"
        with (
            request.urlopen(file_url, timeout=120) as response,
            open(tmp_path, "wb") as output_file,
        ):
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
        except Exception as error:
            api_key = os.getenv("GOOGLE_DRIVE_API_KEY")
            folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
            if api_key and folder_id:
                return api_key, folder_id
            raise ValueError(
                "Missing Google Drive credentials. Configure Databricks secrets "
                "scope='gdrive' keys 'api-key' and 'folder-id', or set env vars "
                "GOOGLE_DRIVE_API_KEY and GOOGLE_DRIVE_FOLDER_ID."
            ) from error

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

            payload = self._drive_get_json("https://www.googleapis.com/drive/v3/files", params)
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

            payload = self._drive_get_json("https://www.googleapis.com/drive/v3/files", params)
            files.extend(payload.get("files", []))

            page_token = payload.get("nextPageToken")
            if page_token is None:
                break

        return files

    # this is a good practice and should be mantained besided the logic "if ID not in
    # table, add to table". Useful when we have multiple documents
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
                start = datetime.datetime.strptime(str(max_processed), "%Y%m%d%H%M").replace(
                    tzinfo=datetime.UTC
                )
                logger.info(
                    "Found existing customs_valuation_metadata table. "
                    f"Starting from: {start.isoformat()}"
                )
                return start

        start = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=5)
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
        end = datetime.datetime.now(datetime.UTC)
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
                    logger.warning(f"Failed to download {filename} for {case_id}: {error}")
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
                        "ingestion_timestamp": datetime.datetime.now(datetime.UTC)
                        .replace(tzinfo=None)
                        .isoformat(),
                        "processed": self.run_processed,
                        "volume_path": case_path,
                    }
                )
            else:
                logger.warning(f"Skipping case {case_id}: missing invoice or declaration PDF.")

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

    # ---------------------------------- Parse Content -----------------------------------
    # this function should be optimized bc ai_parse_document() can be paralelized, and I'm
    # doing it doc by doc.
    def parse_pdfs_with_ai(self) -> None:
        """
        Parse PDFs using ai_parse_document and store in ai_parsed_docs table.

        """

        self.spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {self.parsed_table} (
                path STRING,
                parsed_content STRING,
                processed LONG,
                parsed_content_translated STRING,
                source_language STRING,
                translation_applied BOOLEAN
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
                INSERT INTO {self.parsed_table} (
                    path,
                    parsed_content,
                    processed
                )
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

        logger.info(f"Parsed PDFs for {len(current_case_paths)} cases into {self.parsed_table}")

    # -------------------------------- Translation Logic ---------------------------------
    def translate_parsed_docs(self) -> None:
        """
        Detect source language and translate to English in one LLM call.

        Writes results into:
          - parsed_content_translated
          - source_language
          - translation_applied
        """

        existing_columns = {
            row["col_name"].lower()
            for row in self.spark.sql(f"SHOW COLUMNS IN {self.parsed_table}").collect()
        }

        for column_def in (
            ("parsed_content_translated", "STRING"),
            ("source_language", "STRING"),
            ("translation_applied", "BOOLEAN"),
        ):
            column_name, column_type = column_def
            if column_name.lower() not in existing_columns:
                self.spark.sql(
                    f"ALTER TABLE {self.parsed_table} " f"ADD COLUMN {column_name} {column_type}"
                )

        rows = self.spark.sql(
            f"""
            SELECT path, parsed_content
            FROM {self.parsed_table}
            WHERE processed = {self.run_processed}
                AND parsed_content_translated IS NULL
            """
        ).collect()

        if len(rows) == 0:
            logger.info("No documents to translate for this run.")
            return

        client = mlflow.deployments.get_deploy_client("databricks")
        endpoint_name = self.cfg.llm_endpoint

        translated_rows = []
        for row in rows:
            translated_json, source_language, translation_applied = (
                self._detect_and_translate_document(row["parsed_content"], endpoint_name, client)
            )

            translated_rows.append(
                {
                    "path": row["path"],
                    "parsed_content_translated": translated_json,
                    "source_language": source_language,
                    "translation_applied": translation_applied,
                }
            )

        schema = T.StructType(
            [
                T.StructField("path", T.StringType(), False),
                T.StructField("parsed_content_translated", T.StringType(), True),
                T.StructField("source_language", T.StringType(), True),
                T.StructField("translation_applied", T.BooleanType(), True),
            ]
        )
        translated_df = self.spark.createDataFrame(translated_rows, schema=schema)
        translated_df.createOrReplaceTempView("translation_results")

        self.spark.sql(
            f"""
            MERGE INTO {self.parsed_table} target
            USING translation_results source
            ON target.path = source.path
            WHEN MATCHED THEN UPDATE SET
              target.parsed_content_translated = source.parsed_content_translated,
              target.source_language = source.source_language,
              target.translation_applied = source.translation_applied
            """
        )
        non_english_count = sum(1 for r in translated_rows if r["translation_applied"])
        logger.info(
            f"Analyzed {len(translated_rows)} documents, "
            f"{non_english_count} were non-English and translated to English."
        )

    @staticmethod
    def _parse_llm_translation_json(content: str) -> dict | None:
        """Parse LLM output into a JSON object, supporting common wrappers.

        Attempts in order:
        1) raw content
        2) JSON fenced code blocks
        3) first object-looking `{...}` slice
        """
        text = (content or "").strip()
        if not text:
            return None

        candidates = [text]
        fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        candidates.extend(block.strip() for block in fenced_blocks if block.strip())

        object_match = re.search(r"\{[\s\S]*\}", text)
        if object_match:
            candidates.append(object_match.group(0).strip())

        seen = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                parsed_candidate = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed_candidate, dict):
                return parsed_candidate

        return None

    def _detect_and_translate_document(
        self,
        parsed_content_json: str,
        endpoint_name: str,
        client: object,
    ) -> tuple[str, str, bool]:
        """Detect source language and translate content in one LLM call.

        Sends to LLM text, table, and section_header elements as a JSON array with
        their original idx. Translations are merged back by idx, leaving all
        other element fields and non-translatable elements untouched.

        Returns:
            (translated_json, source_language, translation_applied)
        """
        parsed_dict = json.loads(parsed_content_json)
        elements = parsed_dict.get("document", {}).get("elements", [])

        translatable_items = [
            {
                "idx": idx,
                "type": element.get("type"),
                "content": element.get("content", "").strip(),
            }
            for idx, element in enumerate(elements)
            if element.get("type") in ("text", "table", "section_header")
            and element.get("content", "").strip()
        ]

        if not translatable_items:
            return parsed_content_json, "en", False

        payload = json.dumps(translatable_items, ensure_ascii=False)

        try:
            response = client.predict(
                endpoint=endpoint_name,
                inputs={
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a JSON transformer. Return only one valid JSON "
                                "object that matches the requested schema."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Detect source language and translate content "
                                "to English.\n"
                                "Input is a JSON array with fields idx, type, content.\n"
                                "Output schema (exact keys):\n"
                                '{"source_language":"<iso-639-1>",'
                                '"items":[{"idx":0,"content":"..."}]}\n'
                                "Rules:\n"
                                "- JSON only (no markdown, code fences, or prose).\n"
                                "- Keep one output item per input item, preserving "
                                "the same idx values.\n"
                                "- Translate only content; keep idx unchanged.\n"
                                "- source_language must be lowercase ISO-639-1.\n"
                                "- If source_language is en, keep content unchanged.\n\n"
                                f"{payload}"
                            ),
                        },
                    ]
                },
            )
        except Exception as error:
            error_message = str(error)
            if "does not exist" in error_message.lower():
                raise ValueError(
                    "Configured LLM endpoint does not exist: "
                    f"'{endpoint_name}'. Update 'llm_endpoint' in "
                    "project_config.yml to a valid Model Serving endpoint."
                ) from error
            raise

        content = str(response["choices"][0]["message"].get("content", "")).strip()

        # JSON output tends to be error-prone
        result = self._parse_llm_translation_json(content)

        # in case of JSON output errors
        if result is None:
            preview = content[:300].replace("\n", "\\n")
            logger.warning(
                "Could not parse LLM translation response as JSON. "
                "Returning original document unchanged. Preview: {}",
                preview,
            )
            return parsed_content_json, "en", False

        source_language = str(result.get("source_language", "en")).strip().lower()
        if not source_language:
            source_language = "en"

        # if source language English, skip translation and return original content
        if source_language == "en":
            return parsed_content_json, source_language, False

        translated_items = result.get("items", [])
        translated_by_idx = {}
        for item in translated_items:
            if "idx" not in item or "content" not in item:
                continue
            try:
                item_idx = int(item["idx"])
            except (TypeError, ValueError):
                continue
            translated_by_idx[item_idx] = str(item["content"]).strip()

        if not translated_by_idx:
            logger.warning(
                "LLM translation response had no valid translated items. "
                "Returning original document unchanged."
            )
            return parsed_content_json, source_language, False

        for idx, element in enumerate(elements):
            if idx in translated_by_idx:
                element["content"] = translated_by_idx[idx]

        return json.dumps(parsed_dict), source_language, True

    # ------------------------------- Metadata Extraction --------------------------------
    @staticmethod
    def _extract_invoice_metadata(
        parsed_content_json: str,
        amount_col: int,
        currency_col: int,
    ) -> tuple[float | None, str | None]:
        """From invoice final HTML table last row, read amount/currency by column."""
        if not (1 <= amount_col <= 9 and 1 <= currency_col <= 9):
            return None, None

        parsed_dict = json.loads(parsed_content_json)

        elements = parsed_dict.get("document", {}).get("elements", [])
        table_elements = [
            str(element.get("content", "")).strip()
            for element in elements
            if element.get("type") == "table" and str(element.get("content", "")).strip()
        ]
        if not table_elements:
            return None, None

        final_table_html = table_elements[-1]
        rows = re.findall(
            r"<tr\b[^>]*>([\s\S]*?)</tr>",
            final_table_html,
            flags=re.IGNORECASE,
        )
        if not rows:
            return None, None

        last_row_html = rows[-1]
        raw_cells = re.findall(
            r"<t[dh]\b[^>]*>([\s\S]*?)</t[dh]>",
            last_row_html,
            flags=re.IGNORECASE,
        )
        if not raw_cells:
            return None, None

        cells = [
            re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", cell))).strip()
            for cell in raw_cells
        ]

        try:
            amount = float(cells[amount_col - 1])
            currency = cells[currency_col - 1].upper()
        except Exception as error:
            logger.warning(
                "Failed to extract total/currency with amount_col={} "
                "currency_col={} cells={}. Error: {}",
                amount_col,
                currency_col,
                cells,
                error,
            )
            return None, None

        return amount, currency

    @staticmethod
    def _extract_royalty_metadata(parsed_content_json: str) -> float | None:
        """Extract first percentage value found in content (e.g. 5%, 7,5%)."""
        parsed_dict = json.loads(parsed_content_json)

        elements = parsed_dict.get("document", {}).get("elements", [])
        for element in elements:
            text = str(element.get("content", ""))
            match = re.search(r"(\d+(?:[.,]\d+)?)\s*%", text)
            if match:
                return float(match.group(1).replace(",", "."))

        return None

    @staticmethod
    def _extract_declaration_metadata(parsed_content_json: str) -> dict:
        """Extract declaration values from final table using label-based matching."""
        result = {
            "declaration_invoice_total": None,
            "declaration_tax_A": None,
            "declaration_tax_B": None,
            "declaration_VAT": None,
            "declaration_royalty": None,
            "declaration_total": None,
            "declaration_currency": None,
        }

        parsed_dict = json.loads(parsed_content_json)
        elements = parsed_dict.get("document", {}).get("elements", [])
        table_elements = [
            str(element.get("content", "")).strip()
            for element in elements
            if element.get("type") == "table" and str(element.get("content", "")).strip()
        ]
        if not table_elements:
            return result

        final_table_html = table_elements[-1]
        rows_html = re.findall(
            r"<tr\b[^>]*>([\s\S]*?)</tr>",
            final_table_html,
            flags=re.IGNORECASE,
        )
        if not rows_html:
            return result

        rows = [
            [
                re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", cell))).strip()
                for cell in re.findall(
                    r"<t[dh]\b[^>]*>([\s\S]*?)</t[dh]>",
                    row_html,
                    flags=re.IGNORECASE,
                )
            ]
            for row_html in rows_html
        ]
        rows = [row for row in rows if row]
        if not rows:
            return result

        # Drop header row (row 1) and parse only data rows
        data_rows = rows[1:]

        for row in data_rows:
            if len(row) < 3:
                continue

            label = row[0].strip().lower()
            amount = float(row[2].strip())

            if "invoice" in label:
                result["declaration_invoice_total"] = amount
            elif "tax a" in label:
                result["declaration_tax_A"] = amount
            elif "tax b" in label:
                result["declaration_tax_B"] = amount
            elif "vat" in label:
                result["declaration_VAT"] = amount
            elif "royalty" in label:
                result["declaration_royalty"] = amount
            elif "total" in label:
                result["declaration_total"] = amount

        last_row = rows[-1]
        if len(last_row) >= 4:
            result["declaration_currency"] = last_row[3].strip().upper() or None

        return result

    @staticmethod
    def _extract_structured_metadata(path: str, parsed_content_json: str) -> dict:
        """Extract documents metadata fields by document type."""
        filename = os.path.basename(path).lower()
        result = {
            "invoice_total": None,
            "invoice_currency": None,
            "declaration_invoice_total": None,
            "declaration_tax_A": None,
            "declaration_tax_B": None,
            "declaration_VAT": None,
            "declaration_royalty": None,
            "declaration_total": None,
            "declaration_currency": None,
            "royalty_percentage": None,
        }

        if filename.startswith("invoice_"):
            amount, currency = DataProcessor._extract_invoice_metadata(
                parsed_content_json,
                amount_col=6,
                currency_col=7,
            )
            result["invoice_total"] = amount
            result["invoice_currency"] = currency
        elif filename.startswith("declaration_"):
            result.update(DataProcessor._extract_declaration_metadata(parsed_content_json))
        elif filename.startswith("royalty_"):
            result["royalty_percentage"] = DataProcessor._extract_royalty_metadata(
                parsed_content_json
            )

        return result

    # ------------------------------------- Chunks ---------------------------------------
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
            f"Processing parsed documents from " f"{self.parsed_table} for run {self.run_processed}"
        )

        df = (
            self.spark.table(self.parsed_table)
            .where(f"processed = {self.run_processed}")
            .where(col("parsed_content_translated").isNotNull())
        )

        # Define one base schema used for chunk and structured metadata
        chunk_schema = StructType(
            [
                StructField("chunk_id", StringType(), True),
                StructField("content", StringType(), True),
                StructField("invoice_total", T.DoubleType(), True),
                StructField("invoice_currency", T.StringType(), True),
                StructField("declaration_invoice_total", T.DoubleType(), True),
                StructField("declaration_tax_A", T.DoubleType(), True),
                StructField("declaration_tax_B", T.DoubleType(), True),
                StructField("declaration_VAT", T.DoubleType(), True),
                StructField("declaration_royalty", T.DoubleType(), True),
                StructField("declaration_total", T.DoubleType(), True),
                StructField("declaration_currency", T.StringType(), True),
                StructField("royalty_percentage", T.DoubleType(), True),
            ]
        )

        extract_chunks_udf = udf(
            self._extract_chunks,
            ArrayType(
                StructType(
                    [
                        chunk_schema["chunk_id"],
                        chunk_schema["content"],
                    ]
                )
            ),
        )
        extract_cases_id_udf = udf(self._extract_cases_id, StringType())
        extract_document_id_udf = udf(self._extract_document_id, StringType())
        clean_chunk_udf = udf(self._clean_chunk, StringType())
        extract_structured_metadata_udf = udf(
            self._extract_structured_metadata,
            StructType(
                [
                    chunk_schema["invoice_total"],
                    chunk_schema["invoice_currency"],
                    chunk_schema["declaration_invoice_total"],
                    chunk_schema["declaration_tax_A"],
                    chunk_schema["declaration_tax_B"],
                    chunk_schema["declaration_VAT"],
                    chunk_schema["declaration_royalty"],
                    chunk_schema["declaration_total"],
                    chunk_schema["declaration_currency"],
                    chunk_schema["royalty_percentage"],
                ]
            ),
        )

        metadata_df = self.spark.table(self.docs_table).select(
            col("id").alias("case_id"),
            col("invoice_path"),
            col("declaration_path"),
            col("royalty_path"),
        )

        # groupBy so values propagate to all documents in case_id level
        case_structured_metadata_df = (
            df.withColumn("case_id", extract_cases_id_udf(col("path")))
            .withColumn(
                "structured_metadata",
                extract_structured_metadata_udf(col("path"), col("parsed_content_translated")),
            )
            .groupBy("case_id")
            .agg(
                first(col("structured_metadata.invoice_total"), ignorenulls=True).alias(
                    "invoice_total"
                ),
                first(col("structured_metadata.invoice_currency"), ignorenulls=True).alias(
                    "invoice_currency"
                ),
                first(
                    col("structured_metadata.declaration_invoice_total"),
                    ignorenulls=True,
                ).alias("declaration_invoice_total"),
                first(col("structured_metadata.declaration_tax_A"), ignorenulls=True).alias(
                    "declaration_tax_A"
                ),
                first(col("structured_metadata.declaration_tax_B"), ignorenulls=True).alias(
                    "declaration_tax_B"
                ),
                first(col("structured_metadata.declaration_VAT"), ignorenulls=True).alias(
                    "declaration_VAT"
                ),
                first(
                    col("structured_metadata.declaration_royalty"),
                    ignorenulls=True,
                ).alias("declaration_royalty"),
                first(col("structured_metadata.declaration_total"), ignorenulls=True).alias(
                    "declaration_total"
                ),
                first(
                    col("structured_metadata.declaration_currency"),
                    ignorenulls=True,
                ).alias("declaration_currency"),
                first(col("structured_metadata.royalty_percentage"), ignorenulls=True).alias(
                    "royalty_percentage"
                ),
            )
        )

        # Create the transformed dataframe (parsed_df base table)
        chunks_df = (
            df.withColumn("case_id", extract_cases_id_udf(col("path")))
            .withColumn("document_id", extract_document_id_udf(col("path")))
            .withColumn(
                "chunks",
                extract_chunks_udf(col("parsed_content_translated")),
            )
            .withColumn("chunk", explode(col("chunks")))
            .select(
                col("case_id"),
                col("document_id"),
                col("path").alias("document_path"),
                col("source_language"),
                col("translation_applied"),
                col("chunk.chunk_id").alias("chunk_id"),
                clean_chunk_udf(col("chunk.content")).alias("text"),
                concat_ws(
                    "_",
                    col("case_id"),
                    col("document_id"),
                    col("chunk.chunk_id"),
                ).alias("id"),
            )
            .join(case_structured_metadata_df, "case_id", "left")
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
        chunks_df.write.option("mergeSchema", "true").mode("append").saveAsTable(chunks_table)
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
        Complete workflow: download case PDFs, parse, translate and process chunks.
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

        # [NEW] Step 2b: Detect document language and translate to English
        self.translate_parsed_docs()
        logger.info("Translated documents to English.")

        # Step 3: Process chunks
        self.process_chunks()
        logger.info("Processing complete!")
