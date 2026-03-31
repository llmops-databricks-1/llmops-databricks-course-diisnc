# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.1: Custom Functions & Tools for Agents
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What are agent tools?
# MAGIC - Creating custom functions
# MAGIC - Tool specifications (OpenAI format)
# MAGIC - Integrating tools with agents
# MAGIC - Vector search as a tool


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Understanding Agent Tools
# MAGIC
# MAGIC **Tools** are functions that agents can call to perform specific tasks.
# MAGIC
# MAGIC ### Why Tools?
# MAGIC
# MAGIC LLMs alone cannot:
# MAGIC - Access external data (databases, APIs)
# MAGIC - Perform calculations
# MAGIC - Execute code
# MAGIC - Search documents
# MAGIC
# MAGIC **Tools bridge this gap** by giving LLMs the ability to take actions.
# MAGIC
# MAGIC ### Tool Calling Flow:
# MAGIC
# MAGIC ```
# MAGIC User: "What papers discuss transformers?"
# MAGIC   ↓
# MAGIC Agent: Decides to use vector_search tool
# MAGIC   ↓
# MAGIC Tool: vector_search(query="transformers")
# MAGIC   ↓
# MAGIC Tool Result: [paper1, paper2, paper3]
# MAGIC   ↓
# MAGIC Agent: Synthesizes answer from results
# MAGIC   ↓
# MAGIC Response: "Here are papers about transformers..."
# MAGIC ```

# COMMAND ----------

import json
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from valuation_curator.config import get_env, load_config
from valuation_curator.mcp import ToolInfo

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

# Load configuration
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()

# for development purposes, token with short lifetime
vsc = VectorSearchClient(
    workspace_url=w.config.host,
    personal_access_token=w.tokens.create(lifetime_seconds=1200).token_value,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Tool Specification Format
# MAGIC
# MAGIC Tools are defined using the **OpenAI function calling format**:
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "type": "function",
# MAGIC   "function": {
# MAGIC     "name": "tool_name",
# MAGIC     "description": "What the tool does",
# MAGIC     "parameters": {
# MAGIC       "type": "object",
# MAGIC       "properties": {
# MAGIC         "param1": {
# MAGIC           "type": "string",
# MAGIC           "description": "Description of param1"
# MAGIC         }
# MAGIC       },
# MAGIC       "required": ["param1"]
# MAGIC     }
# MAGIC   }
# MAGIC }
# MAGIC ```
# MAGIC Note:the description is not just for metadata purposes, it's also for the LLM to
# MAGIC understand when to use the tool and how to use it.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Creating a Simple Calculator Tool

# COMMAND ----------


def calculator(operation: str, a: float, b: float) -> float:
    """Perform basic arithmetic operations.

    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First number
        b: Second number

    Returns:
        Result of the operation
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float("inf"),
    }

    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")

    return operations[operation](a, b)


# Test the function
result = calculator("multiply", 5, 3)
logger.info(f"5 * 3 = {result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tool Specification for Calculator

# COMMAND ----------

calculator_tool_spec = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform basic arithmetic operations (add, subtract, multiply, "
        "divide)",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform",
                },
                "a": {"type": "number", "description": "The first number"},
                "b": {"type": "number", "description": "The second number"},
            },
            "required": ["operation", "a", "b"],
        },
    },
}

logger.info("Calculator Tool Specification:")
logger.info(json.dumps(calculator_tool_spec, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Creating a Vector Search Tool
# MAGIC Here, we will create the tool ourselves, but in notebook 3.2 we will see how to
# MAGIC automatically create it with Databricks Vector Search MCP.

# COMMAND ----------


# Helper function to parse vector search results
def parse_vector_search_results(results: dict) -> list[dict]:
    """Parse vector search results from array format to dict format.

    Args:
        results: Raw results from similarity_search()

    Returns:
        List of dictionaries with column names as keys
    """
    columns = [col["name"] for col in results.get("manifest", {}).get("columns", [])]
    data_array = results.get("result", {}).get("data_array", [])

    return [dict(zip(columns, row_data, strict=False)) for row_data in data_array]


# COMMAND ----------


def search_documents(
    query: str, num_results: int = 5, language_filter: str = None
) -> str:
    """Search for relevant papers using vector search.

    Args:
        query: Search query
        num_results: Number of results to return
        language_filter: Optional language filter (e.g., "es")

    Returns:
        JSON string with search results
    """
    index_name = f"{cfg.catalog}.{cfg.schema}.valuation_index"
    index = vsc.get_index(index_name=index_name)

    # Build search parameters
    search_params = {
        "query_text": query,
        "columns": ["text", "id", "case_id", "source_language"],
        "num_results": num_results,
        "query_type": "hybrid",
    }

    # Add language filter if provided
    if language_filter:
        search_params["filters"] = {"source_language": language_filter}

    # Perform search
    results = index.similarity_search(**search_params)

    # Format results using helper function
    papers = []
    for row in parse_vector_search_results(results):
        papers.append(
            {
                "Case": row.get("case_id", "N/A"),
                "ID": row.get("id", "N/A"),
                "language": row.get("source_language", "N/A"),
                "excerpt": row.get("text", "")[:200] + "...",
            }
        )

    return json.dumps(papers, indent=2)


# Test the function
results = search_documents("royalty 3.5%", num_results=2, language_filter="es")
logger.info("Search Results:")
logger.info(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tool Specification for Vector Search

# COMMAND ----------

search_docs_tool_spec = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search for valuation documents using semantic search. Returns "
        "relevant documents with case information, language, and excerpts.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing what documents to find",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5,
                },
                "language_filter": {
                    "type": "string",
                    "description": "Optional language filter to limit results(e.g. 'es')",
                },
            },
            "required": ["query"],
        },
    },
}

logger.info("Search Documents Tool Specification:")
logger.info(json.dumps(search_docs_tool_spec, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Tool Information Class
# MAGIC Now, we turn the functions in a tool by using our ToolInfo helper function.

# COMMAND ----------

# Using ToolInfo from valuation_curator.mcp package
# This class represents a tool with name, spec, and execution function

# Create tool info objects
calculator_tool = ToolInfo(
    name="calculator", spec=calculator_tool_spec, exec_fn=calculator
)

search_documents_tool = ToolInfo(
    name="search_documents", spec=search_docs_tool_spec, exec_fn=search_documents
)

logger.info("Available Tools:")
logger.info(f"1. {calculator_tool.name}")
logger.info(f"2. {search_documents_tool.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Tool Registry Pattern
# MAGIC Example. Not going to be used in the course. Useful if we have many tools and we
# MAGIC want to list them and send to the agent like this:
# MAGIC agent = SimpleAgent(
# MAGIC     llm_endpoint=cfg.llm_endpoint,
# MAGIC     system_prompt="You are a helpful assistant. Use the available tools to answer"
# MAGIC     " questions.",
# MAGIC     tools=registry.get_all_tools())

# COMMAND ----------


class ToolRegistry:
    """Registry for managing agent tools."""

    def __init__(self):
        self._tools: dict[str, ToolInfo] = {}

    def register(self, tool: ToolInfo) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.info(f"✓ Registered tool: {tool.name}")

    def get_tool(self, name: str) -> ToolInfo:
        """Get a tool by name."""
        if name not in self._tools:
            raise ValueError(f"Tool not found: {name}")
        return self._tools[name]

    def get_all_specs(self) -> list[dict]:
        """Get all tool specifications."""
        return [tool.spec for tool in self._tools.values()]

    def execute(self, name: str, args: dict) -> Any:
        """Execute a tool with arguments."""
        tool = self.get_tool(name)
        return tool.exec_fn(**args)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_all_tools(self) -> list[ToolInfo]:
        """Get all tools as a list."""
        return list(self._tools.values())


# Create registry and register tools
registry = ToolRegistry()
registry.register(calculator_tool)
registry.register(search_documents_tool)

logger.info(f"Total tools registered: {len(registry.list_tools())}")
logger.info(f"Tools: {registry.list_tools()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Executing Tools

# COMMAND ----------

# Execute calculator tool
calc_result = registry.execute("calculator", {"operation": "add", "a": 10, "b": 5})
logger.info(f"Calculator result: {calc_result}")

# Execute search tool
search_result = registry.execute(
    "search_documents", {"query": "royalty 3.5%", "num_results": 3}
)
logger.info(f"Search result:\n{search_result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Best Practices for Tool Design
# MAGIC
# MAGIC ### Do:
# MAGIC 1. **Clear descriptions**: Help the LLM understand when to use the tool
# MAGIC 2. **Type hints**: Use proper Python type hints
# MAGIC 3. **Error handling**: Handle errors gracefully
# MAGIC 4. **Return structured data**: JSON or clear text format
# MAGIC 6. **Validate inputs**: Check parameters before execution
# MAGIC 7. **Document parameters**: Clear parameter descriptions
# MAGIC
# MAGIC ### Don't:
# MAGIC 1. Create tools that are too complex
# MAGIC 2. Return unstructured or ambiguous data
# MAGIC 3. Forget error handling
# MAGIC 4. Make tools that take too long to execute
# MAGIC 5. Overlap tool functionality
# MAGIC 6. Use unclear tool names

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Tool Design Patterns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pattern 1: Data Retrieval Tool
# MAGIC ```python
# MAGIC def get_data(query: str) -> str:
# MAGIC     # Fetch data from database/API
# MAGIC     # Format and return
# MAGIC     pass
# MAGIC ```
# MAGIC
# MAGIC ### Pattern 2: Computation Tool
# MAGIC ```python
# MAGIC def calculate(formula: str, values: dict) -> float:
# MAGIC     # Perform calculation
# MAGIC     # Return result
# MAGIC     pass
# MAGIC ```
# MAGIC
# MAGIC ### Pattern 3: Action Tool
# MAGIC ```python
# MAGIC def send_notification(message: str, recipient: str) -> str:
# MAGIC     # Perform action
# MAGIC     # Return confirmation
# MAGIC     pass
# MAGIC ```
# MAGIC
# MAGIC ### Pattern 4: Analysis Tool
# MAGIC ```python
# MAGIC def analyze_data(data: list, metric: str) -> dict:
# MAGIC     # Analyze data
# MAGIC     # Return insights
# MAGIC     pass
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Testing Tools
# MAGIC Since tools are deterministic, we can write tests for inputs, outputs, etc.
# MAGIC This is important before we use the tool in an LLM.

# COMMAND ----------


def test_tool(tool_name: str, test_cases: list[dict]) -> None:
    """Test a tool with multiple test cases."""
    logger.info(f"Testing tool: {tool_name}")
    logger.info("=" * 80)

    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"Test Case {i}:")
        logger.info(f"  Input: {test_case}")

        try:
            result = registry.execute(tool_name, test_case)
            logger.info("  ✓ Success")
            logger.info(f"  Result: {str(result)[:100]}...")
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")


# Test calculator
test_tool(
    "calculator",
    [
        {"operation": "add", "a": 5, "b": 3},
        {"operation": "multiply", "a": 4, "b": 7},
        {"operation": "divide", "a": 10, "b": 2},
    ],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Using Tools with an Agent
# MAGIC
# MAGIC Now let's create a simple agent that can call our tools.
# MAGIC For development purposes only, creating tmp token for openAI API.

# COMMAND ----------


class SimpleAgent:
    """A simple agent that can call tools in a loop."""

    def __init__(self, llm_endpoint: str, system_prompt: str, tools: list[ToolInfo]):
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

    # Summary: get message; check if there are any tool calls; if yes, execute tools and
    # add results to messages; if no, return response
    # max_iterations = max amount of times we try to call a tool and update message
    def chat(self, user_message: str, max_iterations: int = 10) -> str:
        """Chat with the agent, allowing tool calls."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        for _iteration in range(max_iterations):
            # Call LLM (we pass user msg and available tools)
            response = self._client.chat.completions.create(
                model=self.llm_endpoint,
                messages=messages,
                tools=self.get_tool_specs() if self._tools_dict else None,
            )

            # collect response message
            assistant_message = response.choices[0].message

            # Check if LLM wants to call tools (if message has tool names & arguments)
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

                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    logger.info(f"Calling tool: {tool_name}({tool_args})")

                    try:
                        result = self.execute_tool(tool_name, tool_args)
                    except Exception as e:
                        result = f"Error: {str(e)}"

                    # Add tool result to messages
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result),
                        }
                    )
            else:
                # No tool calls, return the response
                return assistant_message.content

        return "Max iterations reached."


# COMMAND ----------


# Create agent with our tools
agent = SimpleAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt="You are a helpful assistant. Use the available tools to answer "
    "questions.",
    tools=[calculator_tool, search_documents_tool],
)

# agent = SimpleAgent(
#     llm_endpoint=cfg.llm_endpoint,
#     system_prompt="You are a helpful assistant. Use the available tools to answer "
#     "questions.",
# tools=registry.get_all_tools())

logger.info("✓ Agent created with tools:")
for tool_name in agent._tools_dict:
    logger.info(f"  - {tool_name}")

# COMMAND ----------

# Test agent with calculator
logger.info("Testing agent with calculator:")
logger.info("=" * 80)

response = agent.chat("What is 42 multiplied by 17?")
logger.info(f"Agent response: {response}")

# COMMAND ----------

# Test agent with search tool
logger.info("Testing agent with search tool:")
logger.info("=" * 80)

response = agent.chat("Find documents with 3.5% royalty")
logger.info(f"Agent response: {response}")
