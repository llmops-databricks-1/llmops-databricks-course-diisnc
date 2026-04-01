# Databricks notebook source
# MAGIC %md
# MAGIC # My agent with MCP tools
# MAGIC
# MAGIC This is a replica of notebook 3.2_mcp_integration.py but adapted to my agent and
# MAGIC the tools it will need:
# MAGIC - Create Vector Search MCP URL (tool) with my index table
# MAGIC - Create Genie Space MCP URL (tool) with my metadata table (initial metadata table
# MAGIC (customs_valuation_metadata) + chunks table (chunks_table) which has more complete
# MAGIC metadata from ingested files)
# MAGIC - Creating others MCP Tools for Agents:
# MAGIC     - anomaly detection (simple logic) using UC functions based on chunks table
# MAGIC       extracted metadata

# COMMAND ----------

import asyncio
import json

import nest_asyncio
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from valuation_curator.config import get_env, load_config
from valuation_curator.mcp import create_mcp_tools

# Enable nested event loops (required for Databricks notebooks)
# (notebooks already have an event loop running in the background)
nest_asyncio.apply()

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

# Load configuration
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Databricks MCP Servers
# MAGIC
# MAGIC Databricks provides managed MCP servers for:
# MAGIC
# MAGIC 1. **Vector Search MCP**: Search vector indexes
# MAGIC 2. **Genie Space MCP**: Query data using natural language
# MAGIC 3. **Unity Catalog Functions MCP**: Execute UC functions
# MAGIC 4. **SQL Warehouse MCP**: Execute SQL queries

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Vector Search MCP

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vector Search MCP URL Format:

# COMMAND ----------

# Create Vector Search MCP URL (matches O'Reilly source)
host = w.config.host
vector_search_mcp_url = f"{host}/api/2.0/mcp/vector-search/{cfg.catalog}/{cfg.schema}"

logger.info("Vector Search MCP URL:")
logger.info(vector_search_mcp_url)

# COMMAND ----------

# MAGIC %md
# MAGIC ### List Available Tools from Vector Search MCP

# COMMAND ----------

# Connect to Vector Search MCP
vs_mcp_client = DatabricksMCPClient(server_url=vector_search_mcp_url, workspace_client=w)

# List available tools
vs_tools = vs_mcp_client.list_tools()

# The tools follow OpenAI specifications so we can extract name, description, ...
logger.info(f"Vector Search MCP Tools ({len(vs_tools)}):")
logger.info("=" * 80)
for tool in vs_tools:
    logger.info(f"Tool: {tool.name}")
    logger.info(f"Description: {tool.description}")
    if tool.inputSchema:
        logger.info(f"Parameters: {list(tool.inputSchema.get('properties', {}).keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Call Vector Search Tool
# MAGIC
# MAGIC **Important**: The MCP tool name uses double underscores:
# MAGIC - Tool name: `workspace__course_data__valuation_index`
# MAGIC - Parameter: `query` (just the search query text)
# MAGIC - The index is already specified in the tool name itself

# COMMAND ----------

# Search for docs with 3.5% royalty
# The tool name is the index name with '__' separators
# NOTE: it's possible to modify the defaults for any tools using fastMCP
#   (e.g. change nº results)
tool_name = f"{cfg.catalog}__{cfg.schema}__valuation_index"

search_result = vs_mcp_client.call_tool(tool_name, {"query": "royalty 3.5%"})

logger.info("Search Results:")
logger.info("=" * 80)
for content in search_result.content:
    logger.info(content.text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Genie Space MCP

# COMMAND ----------

# MAGIC %md
# MAGIC ### Genie Space MCP URL Format:
# MAGIC ```
# MAGIC {workspace_host}/api/2.0/mcp/genie/{genie_space_id}
# MAGIC ```
# MAGIC
# MAGIC **Genie** allows natural language queries over your data.
# MAGIC Note: genie space can be configured in notebook 3.2b or in db UI.
# MAGIC
# MAGIC Genie should be connected to structured data to generate SQL, not embeddings or
# MAGIC vector indexes.

# COMMAND ----------

# Check if Genie space is configured in config file
if hasattr(cfg, "genie_space_id") and cfg.genie_space_id:
    genie_mcp_url = f"{host}/api/2.0/mcp/genie/{cfg.genie_space_id}"
    logger.info("Genie MCP URL:")
    logger.info(genie_mcp_url)

    # Connect to Genie MCP
    genie_mcp_client = DatabricksMCPClient(server_url=genie_mcp_url, workspace_client=w)

    # List available tools
    genie_tools = genie_mcp_client.list_tools()

    logger.info(f"Genie MCP Tools ({len(genie_tools)}):")
    logger.info("=" * 80)
    for tool in genie_tools:
        logger.info(f"Tool: {tool.name}")
        logger.info(f"Description: {tool.description}")
else:
    logger.warning("⚠️ Genie space not configured in project_config.yml")
    logger.info("To use Genie MCP, add 'genie_space_id' to your configuration")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Creating MCP Tools for Agents

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Unity Catalog Function
# MAGIC This needs to be registered as a tool, but MCP server reads the function's
# MAGIC metadata directly from Unity Catalog and auto-generates the OpenAI spec.
# MAGIC Otherwise, the tool won't be called properly.

# COMMAND ----------
print(f"Using catalog: {cfg.catalog}, schema: {cfg.schema}")

# Register anomaly detection function to identify inconsistencies in customs valuation
# data
# note: anomaly_type STRING DEFAULT NULL is needed for tool to automatically define the
# openAI format tool spec with an optional parameter (instead of required)
function_name = f"{cfg.catalog}.{cfg.schema}.detect_anomalies"
spark.sql(
    f"""
    CREATE OR REPLACE FUNCTION {function_name}(
        anomaly_type STRING DEFAULT NULL COMMENT 'Filter by anomaly type: INVOICE_TOTAL_MISMATCH, DECLARATION_TOTAL_MISMATCH, ROYALTY_MISMATCH. Leave empty to return all anomalies.'
    )
    RETURNS TABLE(case_id STRING, anomaly_rule STRING, invoice_total DOUBLE,
    declaration_invoice_total DOUBLE, declaration_total DOUBLE)
    COMMENT 'Detects anomalies in customs valuation cases. Returns cases where invoice totals mismatch, declaration totals do not add up, or royalty percentage is declared but royalty amount is missing.'
    RETURN
    SELECT DISTINCT
        case_id,
        anomaly_rule,
        MAX(invoice_total) as invoice_total,
        MAX(declaration_invoice_total) as declaration_invoice_total,
        MAX(declaration_total) as declaration_total
    FROM (
        -- Rule 1: Invoice totals don't match (invoice vs declaration document)
        SELECT
            case_id,
            'INVOICE_TOTAL_MISMATCH' as anomaly_rule,
            invoice_total,
            declaration_invoice_total,
            declaration_total
        FROM {cfg.full_schema_name}.chunks_table
        WHERE invoice_total IS NOT NULL
          AND declaration_invoice_total IS NOT NULL
          AND invoice_total != declaration_invoice_total

        UNION ALL

        -- Rule 2: Declaration totals don't add up (declaration document)
        SELECT
            case_id,
            'DECLARATION_TOTAL_MISMATCH' as anomaly_rule,
            invoice_total,
            declaration_invoice_total,
            declaration_total
        FROM {cfg.full_schema_name}.chunks_table
        WHERE declaration_invoice_total IS NOT NULL
          AND declaration_total IS NOT NULL
          AND (
              round(
                  declaration_invoice_total + coalesce(declaration_tax_A, 0) +
                  coalesce(declaration_tax_B, 0) + coalesce(declaration_VAT, 0) +
                  coalesce(declaration_royalty, 0),
                  2
              ) != declaration_total
          )

        UNION ALL

        -- Rule 3: Royalty mismatch (royalty vs declaration document)
        SELECT
            case_id,
            'ROYALTY_MISMATCH' as anomaly_rule,
            invoice_total,
            declaration_invoice_total,
            declaration_total
        FROM {cfg.full_schema_name}.chunks_table
        WHERE royalty_percentage IS NOT NULL
          AND declaration_royalty IS NULL
    )
    WHERE anomaly_type IS NULL OR anomaly_rule = anomaly_type
    GROUP BY case_id, anomaly_rule
    ORDER BY case_id, anomaly_rule
    """
)
print(f"Created: {function_name}")

# COMMAND ----------
# Test the function with all anomalies
result = spark.sql(
    f"""
    SELECT * FROM {cfg.catalog}.{cfg.schema}.detect_anomalies(NULL)
    LIMIT 10
"""
)
result.show()

# COMMAND ----------
# Test the function with specific anomaly
result = spark.sql(
    f"""
    SELECT * FROM {cfg.catalog}.{cfg.schema}.detect_anomalies('INVOICE_TOTAL_MISMATCH')
    LIMIT 10
"""
)
result.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load All MCP Tools

# COMMAND ----------

# Define MCP server URLs (combines all MCP URLs: 1 Vector Search, 1 Genie space,
# 1 UC Functions)
mcp_urls = [f"{host}/api/2.0/mcp/vector-search/{cfg.catalog}/{cfg.schema}"]

# Add Genie if configured
if hasattr(cfg, "genie_space_id") and cfg.genie_space_id:
    mcp_urls.append(f"{host}/api/2.0/mcp/genie/{cfg.genie_space_id}")

# Add UC Functions MCP — exposes detect_anomalies (and any other UC functions in schema)
# as deterministic agent tools
uc_functions_mcp_url = f"{host}/api/2.0/mcp/functions/{cfg.catalog}/{cfg.schema}"
mcp_urls.append(uc_functions_mcp_url)

logger.info(f"Loading tools from {len(mcp_urls)} MCP servers...")

# Create tools: combine tools from MCP URLs
mcp_tools = asyncio.run(create_mcp_tools(w, mcp_urls))

logger.info(f"✓ Loaded {len(mcp_tools)} tools from MCP servers")
logger.info("Available Tools:")
for i, tool in enumerate(mcp_tools, 1):
    logger.info(f"{i}. {tool.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Using MCP Tools

# COMMAND ----------

# Create a tools dictionary for easy access
tools_dict = {tool.name: tool for tool in mcp_tools}

# Example: Use vector search tool directly
# The tool name is the index name with '__' separators
vector_search_tool_name = f"{cfg.catalog}__{cfg.schema}__valuation_index"

if vector_search_tool_name in tools_dict:
    search_tool = tools_dict[vector_search_tool_name]

    # Execute the tool - only takes 'query' parameter
    result = search_tool.exec_fn(query="royalty 3.5%")

    logger.info("Search Results:")
    logger.info(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. MCP Tool Specifications (following OpenAI format)

# COMMAND ----------

# View tool specifications (what the LLM sees)
if mcp_tools:
    logger.info("Tool Specifications for LLM:")
    logger.info("=" * 80)

    for tool in mcp_tools:
        logger.info(f"Tool: {tool.name}")
        logger.info(json.dumps(tool.spec, indent=2))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Troubleshooting MCP

# COMMAND ----------


def test_mcp_connection(mcp_url: str) -> bool:
    """Test if MCP server is accessible.

    Args:
        mcp_url: MCP server URL

    Returns:
        True if connection successful
    """
    try:
        client = DatabricksMCPClient(server_url=mcp_url, workspace_client=w)
        tools = client.list_tools()
        logger.info("✓ Connected to MCP server")
        logger.info(f"  URL: {mcp_url}")
        logger.info(f"  Tools available: {len(tools)}")
        return True
    except Exception as e:
        logger.error("✗ Failed to connect to MCP server")
        logger.error(f"  URL: {mcp_url}")
        logger.error(f"  Error: {e}")
        return False


# Test Vector Search MCP
logger.info("Testing Vector Search MCP:")
test_mcp_connection(vector_search_mcp_url)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Using MCP Tools with an Agent
# MAGIC
# MAGIC Now let's create a simple agent that can use MCP tools.

# COMMAND ----------


class SimpleAgent:
    """A simple agent that can call tools in a loop."""

    def __init__(self, llm_endpoint: str, system_prompt: str, tools: list):
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self._tools_dict = {tool.name: tool for tool in tools}
        self._client = OpenAI(
            api_key=w.tokens.create(lifetime_seconds=1200).token_value,
            base_url=f"{w.config.host}/serving-endpoints",
        )

    def get_tool_specs(self) -> list[dict]:
        """Get tool specifications for the LLM."""
        return [tool.spec for tool in self._tools_dict.values()]

    def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool by name."""
        if tool_name not in self._tools_dict:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self._tools_dict[tool_name].exec_fn(**args)

    def chat(self, user_message: str, max_iterations: int = 10) -> str:
        """Chat with the agent, allowing tool calls."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        for _iteration in range(max_iterations):
            response = self._client.chat.completions.create(
                model=self.llm_endpoint,
                messages=messages,
                tools=self.get_tool_specs() if self._tools_dict else None,
            )

            assistant_message = response.choices[0].message

            if assistant_message.tool_calls:
                # Add assistant message with tool calls (exclude unsupported fields)
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in assistant_message.tool_calls
                        ],
                    }
                )

                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    logger.info(f"Calling tool: {tool_name}({tool_args})")

                    try:
                        result = self.execute_tool(tool_name, tool_args)
                    except Exception as e:
                        result = f"Error: {str(e)}"

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result),
                        }
                    )
            else:
                return assistant_message.content

        return "Max iterations reached."


# COMMAND ----------

# Create agent with MCP tools
agent = SimpleAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt="You are a helpful research assistant. Use the available tools to"
    "search for papers and answer questions.",
    tools=mcp_tools,
)

logger.info("✓ Agent created with MCP tools:")
for tool_name in agent._tools_dict:
    logger.info(f"  - {tool_name}")

# COMMAND ----------

# Test agent with MCP vector search tool
# It will call tool: customs__customs_valuation__valuation_index
logger.info("Testing agent with MCP tools:")
logger.info("=" * 80)

response = agent.chat("Find documents with 3.5% royalty.")
logger.info(f"Agent response: {response}")

# COMMAND ----------

# Test agent with metadata it was given
# It will call tool: query_space_..., pool_response_...
logger.info("Testing agent with MCP tools:")
logger.info("=" * 80)

response = agent.chat("What was the date of the last ingested files?")
logger.info(f"Agent response: {response}")

# COMMAND ----------

# Test agent with unity catalog function
# It will call tool: customs__customs_valuation__detect_anomalies(anomaly_type=NULL)
logger.info("Testing agent with MCP tools:")
logger.info("=" * 80)

response = agent.chat("In which cases do I have anomalies?")
logger.info(f"Agent response: {response}")

# COMMAND ----------
# Note: answers can also be accessed in Genie -> monitoring
