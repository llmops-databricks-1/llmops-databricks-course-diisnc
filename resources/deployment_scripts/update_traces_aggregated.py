# Databricks notebook source
# This notebook evaluates unevaluated agent traces and creates an aggregated view
# for monitoring agent performance (latency, token usage, quality scores).

import mlflow
import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.config import ProjectConfig
from arxiv_curator.evaluation import (
    hook_in_post_guideline,
    polite_tone_guideline,
    word_count_check,
)
from arxiv_curator.utils.common import get_widget, set_mlflow_tracking_uri

set_mlflow_tracking_uri()

env = get_widget("env", "dev")
cfg = ProjectConfig.from_yaml("../../project_config.yml", env=env)
mlflow.set_experiment(cfg.experiment_name)

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

catalog = cfg.catalog
schema = cfg.schema

# Source table: raw traces logged by the serving endpoint
traces_table = f"{catalog}.{schema}.trace_logs_2712314122562787"
# Target view: aggregated metrics + quality scores per trace
aggregated_view = f"{catalog}.{schema}.arxiv_traces_aggregated"

# COMMAND ----------
# Get traces not yet evaluated:
# - Filters to our serving endpoint only
# - Excludes traces that already have assessments (i.e., already scored)
# - Parses the JSON response to extract the agent's final text message
#   (skipping tool-call outputs, keeping only type='message')

new_traces_df = spark.sql(f"""
    SELECT
        t.trace_id,
        t.request_preview,
        element_at(
            filter(
                from_json(
                    get_json_object(t.response, '$.output'),
                    'ARRAY<STRUCT<type:STRING, content:ARRAY<STRUCT<text:STRING>>>>'
                ),
                x -> x.type = 'message'
            ),
            1
        ).content[0].text AS response_text
    FROM {traces_table} t
    WHERE tags['model_serving_endpoint_name']
            = 'arxiv-agent-endpoint-dev-course'
      AND (t.assessments IS NULL OR size(t.assessments) = 0)
""")

traces_pdf = new_traces_df.toPandas()
logger.info(f"New traces to evaluate: {len(traces_pdf)}")

# COMMAND ----------
# Build eval input DataFrame in the format mlflow.genai.evaluate expects:
# - inputs: dict with the user query
# - outputs: the agent's response text

eval_pdf = pd.DataFrame(
    {
        "trace_id": traces_pdf["trace_id"],
        "inputs": traces_pdf["request_preview"].apply(lambda x: {"query": x}),
        "outputs": traces_pdf["response_text"],
    }
)

# COMMAND ----------
# Run word_count_check (a cheap heuristic scorer) on ALL traces
# and attach the result back to each trace as MLflow feedback

wc_result = mlflow.genai.evaluate(
    data=eval_pdf[["inputs", "outputs"]],
    scorers=[word_count_check],
)

for trace_id, assessments in zip(
    eval_pdf["trace_id"],
    wc_result.result_df["assessments"],
    strict=True,
):
    val = assessments[0]["feedback"]["value"]
    mlflow.log_feedback(
        trace_id=trace_id,
        name="word_count_check",
        value=val,
    )

logger.info(f"Logged word_count_check for {len(eval_pdf)} traces")

# COMMAND ----------
# Run LLM-judge scorers (polite_tone, hook_in_post) on a 10% sample only
# to control cost — these use an LLM call per trace per scorer

sample_size = max(1, int(len(eval_pdf) * 0.1))
sampled_pdf = eval_pdf.sample(n=sample_size)
logger.info(f"Sampled {len(sampled_pdf)} traces for LLM-judge evaluation")

llm_result = mlflow.genai.evaluate(
    data=sampled_pdf[["inputs", "outputs"]],
    scorers=[polite_tone_guideline, hook_in_post_guideline],
)

for trace_id, assessments in zip(
    sampled_pdf["trace_id"],
    llm_result.result_df["assessments"],
    strict=True,
):
    for a in assessments:
        name = a["assessment_name"]
        val = a["feedback"]["value"]
        mlflow.log_feedback(
            trace_id=trace_id,
            name=name,
            value=val,
        )

logger.info(f"Logged polite_tone/hook_in_post for {len(sampled_pdf)} traces")

# COMMAND ----------
# Create/replace an aggregated SQL view — one clean row per trace for dashboarding.
# It takes raw nested trace data, explodes spans to compute operational metrics,
# reads back quality assessments, and flattens everything.
#
# The view does 4 things per trace:
#
# 1. Basic trace info:
#    - trace_id, request_time, request_preview — straight from the table
#    - response_text — parses JSON response to extract the agent's final message
#    - latency_seconds — converts execution_duration_ms to seconds
#
# 2. Span-level metrics (why we need LATERAL VIEW explode + GROUP BY):
#    - LATERAL VIEW explode(spans) flattens the spans array into rows
#    - call_llm_exec_count — counts spans named 'call_llm'
#    - tool_call_count — counts spans named 'execute_tool'
#    - total_tokens_used — sums total_tokens from each call_llm span's output
#    - GROUP BY collapses the exploded rows back to one row per trace
#
# 3. Assessment scores (reading back the feedback we logged earlier):
#    - word_count_check — 1 if 'true', else 0
#    - polite_tone — 1 if 'Pass', else 0
#    - hook_in_post — 1 if 'Pass', else 0
#    - Uses try_element_at (not element_at) so it returns NULL instead of
#      erroring when the assessment doesn't exist (e.g., traces not in
#      the 10% LLM-judge sample)
#
# 4. Metadata:
#    - processed_ts — current_timestamp(), stamps when the view was queried

spark.sql(f"""
    CREATE OR REPLACE VIEW {aggregated_view} AS
    SELECT
        t.trace_id,
        t.request_time,
        t.request_preview,
        element_at(
            filter(
                from_json(
                    get_json_object(t.response, '$.output'),
                    'ARRAY<STRUCT<type:STRING, content:ARRAY<STRUCT<text:STRING>>>>'
                ),
                x -> x.type = 'message'
            ),
            1
        ).content[0].text AS response_text,
        CAST(t.execution_duration_ms / 1000.0 AS DOUBLE)
            AS latency_seconds,
        COUNT(IF(s.name = 'call_llm', 1, NULL))
            AS call_llm_exec_count,
        COUNT(IF(s.name = 'execute_tool', 1, NULL))
            AS tool_call_count,
        CAST(SUM(
            IF(
                s.name = 'call_llm',
                CAST(
                    get_json_object(
                        get_json_object(
                            s.attributes['mlflow.spanOutputs'],
                            '$.usage'
                        ),
                        '$.total_tokens'
                    ) AS INT
                ),
                0
            )
        ) AS LONG) AS total_tokens_used,
        current_timestamp() AS processed_ts,
        CASE
            WHEN try_element_at(
                filter(t.assessments, a -> a.name = 'word_count_check'),
                1
            ).feedback.value = 'true' THEN 1 ELSE 0
        END AS word_count_check,
        CASE
            WHEN try_element_at(
                filter(t.assessments, a -> a.name = 'polite_tone'),
                1
            ).feedback.value = 'Pass' THEN 1 ELSE 0
        END AS polite_tone,
        CASE
            WHEN try_element_at(
                filter(t.assessments, a -> a.name = 'hook_in_post'),
                1
            ).feedback.value = 'Pass' THEN 1 ELSE 0
        END AS hook_in_post
    FROM {traces_table} t
    LATERAL VIEW explode(spans) AS s
    WHERE tags['model_serving_endpoint_name']
            = 'arxiv-agent-endpoint-dev-course'
    GROUP BY t.trace_id, t.request_time,
             t.execution_duration_ms, t.request_preview,
             t.response, t.assessments
""")

logger.info(f"View {aggregated_view} created")

# COMMAND ----------
