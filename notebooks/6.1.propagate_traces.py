# Databricks notebook source
import random
import time
from datetime import datetime

from databricks.sdk import WorkspaceClient
from openai import OpenAI

workspace = WorkspaceClient()
host = workspace.config.host
token = workspace.tokens.create(lifetime_seconds=2000).token_value

endpoint_name = "valuation-agent-endpoint-dev-course"

client = OpenAI(
    api_key=token,
    base_url=f"{host}/serving-endpoints",
)

# COMMAND ----------
# helper notebook to send requests to the endpoint to generate traces
queries = [
    "What are the royalty rates mentioned in the documents?",
    "Which suppliers are referenced across all documents?",
    "List documents with royalty percentages above 5%.",
    "In which documents do I have anomalies?",
    "Give me an overview of possible anomalies.",
    "How many documents are in Spanish?",
    "Are there any invoices from supplier John Doe?",
    "Which document has the lowest royalty rate?",
    "What royalty rates are between 2% and 4%?",
    "Find documents mentioning leather products.",
    "Which suppliers have royalty rates over 10%?",
    "Show documents ingested in the last month.",
    "Find suppliers with documents in multiple languages.",
]

# COMMAND ----------
# for each query
# we send request one by one and we get agent response. In agent, we have mlflow traces
# enabled, so every request generates a trace in our mlflow experiment.
for i, query in enumerate(queries):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
    request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

    print(f"[{i + 1}/30] {query[:60]}...")
    response = client.responses.create(
        model=endpoint_name,
        input=[
            {"role": "user", "content": query}
        ],
        extra_body={"custom_inputs": {
            "session_id": session_id,
            "request_id": request_id,
        }},
    )
    time.sleep(2)

# COMMAND ----------
