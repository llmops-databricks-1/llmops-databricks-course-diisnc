# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.5: Simple RAG with Vector Search
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What is RAG (Retrieval-Augmented Generation)?
# MAGIC - Using Vector Search for document retrieval
# MAGIC - Enriching prompts with retrieved context
# MAGIC - Building a simple Q&A system
# MAGIC
# MAGIC **RAG Flow:**
# MAGIC ```
# MAGIC User Question
# MAGIC     ↓
# MAGIC Vector Search (retrieve relevant documents)
# MAGIC     ↓
# MAGIC Build Prompt (question + context)
# MAGIC     ↓
# MAGIC LLM (generate answer)
# MAGIC     ↓
# MAGIC Response
# MAGIC ```

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from valuation_curator.config import get_env, load_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()

# Create OpenAI client for Databricks
client = OpenAI(
    api_key=w.tokens.create(lifetime_seconds=1200).token_value,
    base_url=f"{w.config.host}/serving-endpoints",
)

# Create Vector Search client
vsc = VectorSearchClient(
    workspace_url=w.config.host,
    personal_access_token=w.tokens.create(lifetime_seconds=1200).token_value,
)

logger.info(f"✓ Connected to workspace: {w.config.host}")
logger.info(f"✓ Using LLM endpoint: {cfg.llm_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Vector Search Retrieval
# MAGIC
# MAGIC First, let's create a function to retrieve relevant documents from our vector
# MAGIC search index.

# COMMAND ----------


def retrieve_documents(query: str, num_results: int = 5) -> list[dict]:
    """Retrieve relevant documents from vector search.

    Args:
        query: The search query
        num_results: Number of documents to retrieve

    Returns:
        List of document dictionaries with title, text, and metadata
    """
    index_name = f"{cfg.catalog}.{cfg.schema}.valuation_index"
    index = vsc.get_index(index_name=index_name)

    results = index.similarity_search(
        query_text=query,
        columns=["text", "id", "case_id", "source_language"],
        num_results=num_results,
        query_type="hybrid",
    )

    # Parse results
    documents = []
    if results and "result" in results:
        data_array = results["result"].get("data_array", [])
        for row in data_array:
            documents.append(
                {"text": row[0], "ID": row[1], "Case ID": row[2], "Language": row[3]}
            )

    return documents


# COMMAND ----------


# Test retrieval
query = "royalty 3.5%"
docs = retrieve_documents(query, num_results=3)

logger.info(f"Retrieved {len(docs)} documents for query: '{query}'")
for i, doc in enumerate(docs, 1):
    logger.info(f"\n{i}. ID: {doc['ID']}")
    logger.info(f"   Case ID: {doc['Case ID']}")
    logger.info(f"   Text preview: {doc['text'][:150]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Building the RAG Prompt
# MAGIC
# MAGIC Now let's create a function that builds a prompt with the retrieved context.

# COMMAND ----------


# the docs will be the documents retrieved in the retrieve_documents() function
# the prompt instruction "Cite the relevant Case ID when making claims" is a good practice
# to encourage the model to reference the source of its information
def build_rag_prompt(question: str, documents: list[dict]) -> str:
    """Build a prompt with retrieved context.

    Args:
        question: The user's question
        documents: List of retrieved documents

    Returns:
        Formatted prompt string
    """
    # Format context from documents
    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_parts.append(f"""
                            ID {i}: {doc["ID"]}
                            Case ID: {doc["Case ID"]}
                            Content: {doc["text"]}
                            """)

    context = "\n---\n".join(context_parts)

    prompt = f"""You are a helpful research assistant. Answer the question based on the
    provided context from research papers.

    CONTEXT:
    {context}

    QUESTION: {question}

    INSTRUCTIONS:
    - Answer based on the provided context
    - If the context doesn't contain enough information, say so
    - Cite the relevant Case ID when making claims
    - Be concise but thorough

    ANSWER:"""

    return prompt


# COMMAND ----------


# Test prompt building
test_prompt = build_rag_prompt("What documents have a 3.5% royalty?", docs)
logger.info("Built RAG prompt:")
logger.info(f"Prompt length: {len(test_prompt)} characters")
logger.info(f"Preview:\n{test_prompt[:500]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. RAG Query Function
# MAGIC
# MAGIC Combine retrieval and generation into a single function.

# COMMAND ----------


def rag_query(question: str, num_docs: int = 5) -> dict:
    """Answer a question using RAG.

    Args:
        question: The user's question
        num_docs: Number of documents to retrieve

    Returns:
        Dictionary with answer and sources
    """
    # Step 1: Retrieve relevant documents
    logger.info(f"Retrieving documents for: '{question}'")
    documents = retrieve_documents(question, num_results=num_docs)
    logger.info(f"Retrieved {len(documents)} documents")

    # Step 2: Build prompt with context
    prompt = build_rag_prompt(question, documents)

    # Step 3: Generate answer with LLM
    logger.info("Generating answer...")
    response = client.chat.completions.create(
        model=cfg.llm_endpoint,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.7,
    )

    answer = response.choices[0].message.content

    # Return answer with sources
    return {
        "question": question,
        "answer": answer,
        "sources": [{"ID": doc["ID"], "Case ID": doc["Case ID"]} for doc in documents],
    }


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test RAG System

# COMMAND ----------


# Test with a research question
result = rag_query("What documents have a 3.5% royalty?")

logger.info("=" * 80)
logger.info(f"Question: {result['question']}")
logger.info("=" * 80)
logger.info(f"\nAnswer:\n{result['answer']}")
logger.info("\nSources:")
for src in result["sources"]:
    logger.info(f"  - ID: {src['ID']}, Case ID: {src['Case ID']}")

# COMMAND ----------

# Test with another question
result2 = rag_query("Do I have any invoice from a supplier called Elena Silva?")

logger.info("=" * 80)
logger.info(f"Question: {result2['question']}")
logger.info("=" * 80)
logger.info(f"\nAnswer:\n{result2['answer']}")
logger.info("\nSources:")
for src in result2["sources"]:
    logger.info(f"  - ID: {src['ID']}, Case ID: {src['Case ID']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. RAG with Conversation History
# MAGIC
# MAGIC Extend RAG to support multi-turn conversations.
# MAGIC It's possible to limit context windows by summarizing previous contexts,
# MAGIC only keep last n prompts, etc

# COMMAND ----------


class SimpleRAG:
    """Simple RAG system with conversation history."""

    def __init__(self, llm_endpoint: str, index_name: str):
        self.llm_endpoint = llm_endpoint
        self.index_name = index_name
        self.conversation_history = []

        # Initialize clients
        self.w = WorkspaceClient()
        self.client = OpenAI(
            api_key=self.w.tokens.create(lifetime_seconds=1200).token_value,
            base_url=f"{self.w.config.host}/serving-endpoints",
        )
        self.vsc = VectorSearchClient(
            workspace_url=self.w.config.host,
            personal_access_token=self.w.tokens.create(lifetime_seconds=1200).token_value,
        )

    def retrieve(self, query: str, num_results: int = 5) -> list[dict]:
        """Retrieve relevant documents."""
        index = self.vsc.get_index(index_name=self.index_name)
        results = index.similarity_search(
            query_text=query,
            columns=["text", "id", "case_id"],
            num_results=num_results,
            query_type="hybrid",
        )

        documents = []
        if results and "result" in results:
            for row in results["result"].get("data_array", []):
                documents.append(
                    {
                        "text": row[0],
                        "ID": row[1],
                        "Case ID": row[2],
                    }
                )
        return documents

    def chat(self, question: str, num_docs: int = 3) -> str:
        """Chat with RAG, maintaining conversation history.
        Similar to previous functionality, but we pass the history also:
            - self.conversation_history = []
        """
        # Retrieve documents
        documents = self.retrieve(question, num_results=num_docs)

        # Build context
        context = "\n\n".join(
            [
                f"[ID: {doc['ID']}, Case ID: {doc['Case ID']}]: {doc['text']}"
                for doc in documents
            ]
        )

        # Build system message with context
        system_message = f"""You are a helpful research assistant. Use the following
        context from research papers to answer questions.
        CONTEXT:
        {context}
        If the context doesn't contain relevant information, say so.
        Cite the relevant Case ID when making claims."""

        # Add user message to history
        self.conversation_history.append({"role": "user", "content": question})

        # Build messages for LLM
        messages = [
            {"role": "system", "content": system_message}
        ] + self.conversation_history

        # Generate response
        response = self.client.chat.completions.create(
            model=self.llm_endpoint,
            messages=messages,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content

        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": answer})

        return answer

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []


# COMMAND ----------


# Create RAG instance
index_name = f"{cfg.catalog}.{cfg.schema}.valuation_index"
rag = SimpleRAG(llm_endpoint=cfg.llm_endpoint, index_name=index_name)

logger.info("✓ SimpleRAG initialized")

# COMMAND ----------

# Multi-turn conversation
logger.info("Starting multi-turn RAG conversation...")
logger.info("=" * 80)

# First question
q1 = "Do I have any invoice from a supplier called Elena Silva?"
a1 = rag.chat(q1)
logger.info(f"Q: {q1}")
logger.info(f"A: {a1}\n")

# rag.conversation_history

# COMMAND ----------

# Follow-up question (uses conversation history)
# What is the total of that invoice? -> does not work well with semantic search, needs
# some field lookup tool sent to an agent
q2 = "And do I have any invoice from Iberia Wholesale?"
a2 = rag.chat(q2)
logger.info(f"Q: {q2}")
logger.info(f"A: {a2}\n")

# COMMAND ----------

# Another follow-up
q3 = "In which documents do we have products like leather handbags?"
a3 = rag.chat(q3)
logger.info(f"Q: {q3}")
logger.info(f"A: {a3}")

# rag.conversation_history
