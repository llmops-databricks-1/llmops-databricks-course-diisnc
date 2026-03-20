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
from typing import Any, Callable
from pydantic import BaseModel
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
import mlflow
from mlflow.entities import SpanType
from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.config import load_config, get_env
from arxiv_curator.mcp import ToolInfo

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

# Load configuration
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()
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
        'add': lambda x, y: x + y,
        'subtract': lambda x, y: x - y,
        'multiply': lambda x, y: x * y,
        'divide': lambda x, y: x / y if y != 0 else float('inf')
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
        "description": "Perform basic arithmetic operations (add, subtract, multiply, divide)",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "The first number"
                },
                "b": {
                    "type": "number",
                    "description": "The second number"
                }
            },
            "required": ["operation", "a", "b"]
        }
    }
}

logger.info("Calculator Tool Specification:")
logger.info(json.dumps(calculator_tool_spec, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Creating a Vector Search Tool

# COMMAND ----------

# Helper function to parse vector search results
def parse_vector_search_results(results):
    """Parse vector search results from array format to dict format.
    
    Args:
        results: Raw results from similarity_search()
        
    Returns:
        List of dictionaries with column names as keys
    """
    columns = [col['name'] for col in results.get('manifest', {}).get('columns', [])]
    data_array = results.get('result', {}).get('data_array', [])
    
    return [dict(zip(columns, row_data)) for row_data in data_array]

# COMMAND ----------

@mlflow.trace(span_type=SpanType.TOOL)
def search_papers(query: str, num_results: int = 5, year_filter: str = None) -> str:
    """Search for relevant papers using vector search.
    
    Args:
        query: Search query
        num_results: Number of results to return
        year_filter: Optional year filter (e.g., "2024")
        
    Returns:
        JSON string with search results
    """
    index_name = f"{cfg.catalog}.{cfg.schema}.arxiv_index"
    index = vsc.get_index(index_name=index_name)
    
    # Build search parameters
    search_params = {
        "query_text": query,
        "columns": ["text", "title", "arxiv_id", "authors", "year"],
        "num_results": num_results,
        "query_type": "hybrid"
    }
    
    # Add year filter if provided
    if year_filter:
        search_params["filters"] = {"year": year_filter}
    
    # Perform search
    results = index.similarity_search(**search_params)
    
    # Format results using helper function
    papers = []
    for row in parse_vector_search_results(results):
        papers.append({
            "title": row.get("title", "N/A"),
            "arxiv_id": row.get("arxiv_id", "N/A"),
            "authors": str(row.get("authors", "N/A")),
            "year": row.get("year", "N/A"),
            "excerpt": row.get("text", "")[:200] + "..."
        })
    
    return json.dumps(papers, indent=2)

# Test the function
results = search_papers("machine learning", num_results=2)
logger.info("Search Results:")
logger.info(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tool Specification for Vector Search

# COMMAND ----------

search_papers_tool_spec = {
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": "Search for academic papers using semantic search. Returns relevant papers with titles, authors, and excerpts.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing what papers to find"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5
                },
                "year_filter": {
                    "type": "string",
                    "description": "Optional year filter to limit results (e.g., '2024')"
                }
            },
            "required": ["query"]
        }
    }
}

logger.info("Search Papers Tool Specification:")
logger.info(json.dumps(search_papers_tool_spec, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Tool Information Class

# COMMAND ----------

# Using ToolInfo from arxiv_curator.mcp package
# This class represents a tool with name, spec, and execution function

# Create tool info objects
calculator_tool = ToolInfo(
    name="calculator",
    spec=calculator_tool_spec,
    exec_fn=calculator
)

search_papers_tool = ToolInfo(
    name="search_papers",
    spec=search_papers_tool_spec,
    exec_fn=search_papers
)

logger.info("Available Tools:")
logger.info(f"1. {calculator_tool.name}")
logger.info(f"2. {search_papers_tool.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Tool Registry Pattern

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

# Create registry and register tools
registry = ToolRegistry()
registry.register(calculator_tool)
registry.register(search_papers_tool)

logger.info(f"Total tools registered: {len(registry.list_tools())}")
logger.info(f"Tools: {registry.list_tools()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Executing Tools

# COMMAND ----------

# Execute calculator tool
calc_result = registry.execute("calculator", {
    "operation": "add",
    "a": 10,
    "b": 5
})
logger.info(f"Calculator result: {calc_result}")

# Execute search tool
search_result = registry.execute("search_papers", {
    "query": "neural networks",
    "num_results": 3
})
logger.info(f"Search result:\n{search_result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Best Practices for Tool Design
# MAGIC
# MAGIC ### ✅ Do:
# MAGIC 1. **Clear descriptions**: Help the LLM understand when to use the tool
# MAGIC 2. **Type hints**: Use proper Python type hints
# MAGIC 3. **Error handling**: Handle errors gracefully
# MAGIC 4. **Return structured data**: JSON or clear text format
# MAGIC 5. **Add tracing**: Use `@mlflow.trace` for observability
# MAGIC 6. **Validate inputs**: Check parameters before execution
# MAGIC 7. **Document parameters**: Clear parameter descriptions
# MAGIC
# MAGIC ### ❌ Don't:
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

# COMMAND ----------

def test_tool(tool_name: str, test_cases: list[dict]):
    """Test a tool with multiple test cases."""
    logger.info(f"Testing tool: {tool_name}")
    logger.info("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"Test Case {i}:")
        logger.info(f"  Input: {test_case}")
        
        try:
            result = registry.execute(tool_name, test_case)
            logger.info(f"  ✓ Success")
            logger.info(f"  Result: {str(result)[:100]}...")
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")

# Test calculator
test_tool("calculator", [
    {"operation": "add", "a": 5, "b": 3},
    {"operation": "multiply", "a": 4, "b": 7},
    {"operation": "divide", "a": 10, "b": 2},
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Using Tools with an Agent
# MAGIC
# MAGIC Now let's use these tools with an actual LLM agent using `SimpleAgent` from our package.

# COMMAND ----------

from arxiv_curator.agent import SimpleAgent
from mlflow.types.responses import ResponsesAgentRequest

# Create agent with our tools
agent = SimpleAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt="You are a helpful assistant. Use the available tools to answer questions.",
    tools=[calculator_tool, search_papers_tool]
)

logger.info("✓ Agent created with tools:")
for tool_name in agent._tools_dict.keys():
    logger.info(f"  - {tool_name}")

# COMMAND ----------

# Test agent with calculator
logger.info("Testing agent with calculator:")
logger.info("=" * 80)

response = agent.predict(ResponsesAgentRequest(
    input=[{"role": "user", "content": "What is 42 multiplied by 17?"}]
))

logger.info(f"Agent response: {response.output[-1].content}")

# COMMAND ----------

# Test agent with search tool
logger.info("Testing agent with search tool:")
logger.info("=" * 80)

response = agent.predict(ResponsesAgentRequest(
    input=[{"role": "user", "content": "Find papers about attention mechanisms"}]
))

logger.info(f"Agent response: {response.output[-1].content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Agent Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────┐
# MAGIC │           User Request                   │
# MAGIC └──────────────┬──────────────────────────┘
# MAGIC                │
# MAGIC                ↓
# MAGIC ┌─────────────────────────────────────────┐
# MAGIC │           Agent                          │
# MAGIC │  - System Prompt                         │
# MAGIC │  - Conversation History                  │
# MAGIC │  - Tool Registry                         │
# MAGIC └──────────────┬──────────────────────────┘
# MAGIC                │
# MAGIC                ↓
# MAGIC ┌─────────────────────────────────────────┐
# MAGIC │           LLM                            │
# MAGIC │  - Decides: Answer or Use Tool?         │
# MAGIC └──────────────┬──────────────────────────┘
# MAGIC                │
# MAGIC        ┌───────┴───────┐
# MAGIC        │               │
# MAGIC        ↓               ↓
# MAGIC   Direct Answer    Tool Call
# MAGIC        │               │
# MAGIC        │               ↓
# MAGIC        │      ┌─────────────────┐
# MAGIC        │      │  Execute Tool   │
# MAGIC        │      └────────┬────────┘
# MAGIC        │               │
# MAGIC        │               ↓
# MAGIC        │      ┌─────────────────┐
# MAGIC        │      │  Tool Result    │
# MAGIC        │      └────────┬────────┘
# MAGIC        │               │
# MAGIC        │               ↓
# MAGIC        │      ┌─────────────────┐
# MAGIC        │      │  LLM (again)    │
# MAGIC        │      │  Synthesize     │
# MAGIC        │      └────────┬────────┘
# MAGIC        │               │
# MAGIC        └───────┬───────┘
# MAGIC                │
# MAGIC                ↓
# MAGIC ┌─────────────────────────────────────────┐
# MAGIC │           Final Response                 │
# MAGIC └─────────────────────────────────────────┘
# MAGIC ```