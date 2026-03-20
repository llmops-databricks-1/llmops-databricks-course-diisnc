# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 3.4: Session Memory with Lakebase
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Lakebase (Databricks PostgreSQL) for session persistence
# MAGIC - Managing conversation history
# MAGIC - Connection pooling and authentication
# MAGIC - Building stateful agents

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import DatabaseInstance
from uuid import uuid4
from loguru import logger

from arxiv_curator.memory import LakebaseMemory

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create Lakebase Instance
# MAGIC
# MAGIC **Lakebase** is Databricks' managed PostgreSQL service:
# MAGIC - Fully managed and serverless
# MAGIC - Integrated with Databricks authentication
# MAGIC - Supports standard PostgreSQL features
# MAGIC - Ideal for session state, caching, and metadata

# COMMAND ----------

w = WorkspaceClient()

instance_name = "arxiv-agent-instance"

# Create or get existing instance
try:
    instance = w.database.get_database_instance(instance_name)
    logger.info(f"Using existing instance: {instance_name}")
    lakebase_host = instance.read_write_dns
except Exception:
    logger.info(f"Creating new instance: {instance_name}")
    instance = w.database.create_database_instance(
        DatabaseInstance(name=instance_name, capacity="CU_1")
    )
    lakebase_host = instance.response.read_write_dns

logger.info(f"Lakebase host: {lakebase_host}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize Memory Manager
# MAGIC
# MAGIC The `LakebaseMemory` class handles:
# MAGIC - Connection pooling
# MAGIC - Authentication (SPN or user credentials)
# MAGIC - Table creation
# MAGIC - Message persistence

# COMMAND ----------

memory = LakebaseMemory(
    host=lakebase_host,
    instance_name=instance_name,
)

logger.info("✓ Memory manager initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Save and Load Messages
# MAGIC
# MAGIC Messages are stored per session ID:
# MAGIC - Each session has a unique ID
# MAGIC - Messages are stored in order
# MAGIC - Sessions can be resumed later

# COMMAND ----------

# Create a test session
session_id = f"test-session-{uuid4()}"

# Save some messages
test_messages = [
    {"role": "user", "content": "What are recent papers on transformers?"},
    {"role": "assistant", "content": "Here are some recent papers on transformer architectures..."},
    {"role": "user", "content": "Tell me more about the first one"},
]

memory.save_messages(session_id, test_messages)
logger.info(f"✓ Saved {len(test_messages)} messages to session: {session_id}")

# COMMAND ----------

# Load messages back
loaded_messages = memory.load_messages(session_id)

logger.info(f"✓ Loaded {len(loaded_messages)} messages:")
for i, msg in enumerate(loaded_messages, 1):
    logger.info(f"  {i}. [{msg['role']}] {msg['content'][:50]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Multi-Turn Conversation
# MAGIC
# MAGIC Demonstrate a stateful conversation:

# COMMAND ----------

# Start a new session
conversation_id = f"conversation-{uuid4()}"

# Turn 1
turn1_messages = [
    {"role": "user", "content": "I'm interested in LLM evaluation metrics"}
]
memory.save_messages(conversation_id, turn1_messages)

# Simulate agent response
turn1_response = [
    {"role": "assistant", "content": "Common LLM evaluation metrics include BLEU, ROUGE, and BERTScore..."}
]
memory.save_messages(conversation_id, turn1_response)

# Turn 2 - reference to previous context
turn2_messages = [
    {"role": "user", "content": "Which one is best for summarization?"}
]
memory.save_messages(conversation_id, turn2_messages)

# Load full conversation
full_conversation = memory.load_messages(conversation_id)

logger.info(f"✓ Full conversation ({len(full_conversation)} messages):")
for msg in full_conversation:
    logger.info(f"  [{msg['role']}] {msg['content']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Using with ArxivAgent
# MAGIC
# MAGIC Integrate memory with the agent:

# COMMAND ----------

from arxiv_curator.agent import ArxivAgent
from arxiv_curator.config import load_config, get_env

# Load config
env = get_env()
cfg = load_config("../project_config.yml", env)

# Create agent
agent = ArxivAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=cfg.system_prompt,
    catalog=cfg.catalog,
    schema=cfg.schema,
    genie_space_id=cfg.genie_space_id,
)

logger.info("✓ Agent created")

# COMMAND ----------

# Create a new session with memory
agent_session_id = f"agent-session-{uuid4()}"

# First query
query1 = {"input": [{"role": "user", "content": "Find papers about RAG"}]}
response1 = agent.predict(query1)

# Save to memory
memory.save_messages(agent_session_id, [
    {"role": "user", "content": "Find papers about RAG"},
    {"role": "assistant", "content": response1["output"][0]["content"]},
])

logger.info("✓ First query completed and saved")

# COMMAND ----------

# Follow-up query with context
# Load previous messages
previous_messages = memory.load_messages(agent_session_id)

# Add new query
query2 = {
    "input": previous_messages + [
        {"role": "user", "content": "What about the most cited one?"}
    ]
}

response2 = agent.predict(query2)

# Save new turn
memory.save_messages(agent_session_id, [
    {"role": "user", "content": "What about the most cited one?"},
    {"role": "assistant", "content": response2["output"][0]["content"]},
])

logger.info("✓ Follow-up query completed with context")

# COMMAND ----------

# View full conversation
full_agent_conversation = memory.load_messages(agent_session_id)

logger.info(f"✓ Full agent conversation ({len(full_agent_conversation)} messages):")
for i, msg in enumerate(full_agent_conversation, 1):
    role = msg["role"]
    content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
    logger.info(f"  {i}. [{role}] {content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook, we learned:
# MAGIC
# MAGIC 1. ✅ How to create and manage Lakebase instances
# MAGIC 2. ✅ How to use `LakebaseMemory` for session persistence
# MAGIC 3. ✅ How to save and load conversation history
# MAGIC 4. ✅ How to build stateful multi-turn conversations
# MAGIC 5. ✅ How to integrate memory with agents
# MAGIC
# MAGIC **Next Steps:**
# MAGIC - Implement session expiration
# MAGIC - Add conversation summarization
# MAGIC - Build a chatbot UI with session management
