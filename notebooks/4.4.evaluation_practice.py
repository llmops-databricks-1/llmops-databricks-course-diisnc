# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 4.5: Agent Evaluation Practice
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Creating evaluation datasets
# MAGIC - Evaluating agents end-to-end
# MAGIC - Domain-specific scorers
# MAGIC - Analyzing evaluation results
# MAGIC - Iterating based on results
# MAGIC - Production evaluation pipelines


# COMMAND ----------

import mlflow
from datetime import datetime
from mlflow.types.responses import ResponsesAgentRequest
from mlflow.genai.judges import make_judge
from pyspark.sql import SparkSession
from loguru import logger
from arxiv_curator.config import load_config, get_env
from arxiv_curator.evaluation import (
    polite_tone_guideline,
    hook_in_post_guideline,
    scope_guideline,
    word_count_check,
    mentions_papers
)
from arxiv_curator.agent import ArxivAgent

# COMMAND ----------

# Setup
mlflow.set_tracking_uri("databricks")

spark = SparkSession.builder.getOrCreate()

# Load configuration
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

# Set experiment
mlflow.set_experiment(cfg.experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create Evaluation Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluation Dataset Structure:
# MAGIC
# MAGIC ```python
# MAGIC eval_data = [
# MAGIC     {
# MAGIC         "inputs": {"question": "User question"},
# MAGIC         "expected_output": "Optional expected answer",  # For reference
# MAGIC         "metadata": {"category": "search", "difficulty": "easy"}
# MAGIC     },
# MAGIC     ...
# MAGIC ]
# MAGIC ```

# COMMAND ----------

# Create evaluation dataset for research assistant agent
eval_data = [
    # Search queries
    {
        "inputs": {"question": "What are recent papers about transformers?"},
        "metadata": {"category": "search", "difficulty": "easy"}
    },
    {
        "inputs": {"question": "Find papers on attention mechanisms published in 2024"},
        "metadata": {"category": "search_filtered", "difficulty": "medium"}
    },
    {
        "inputs": {"question": "What papers discuss the relationship between model size and performance in LLMs?"},
        "metadata": {"category": "search_complex", "difficulty": "hard"}
    },
    
    # Analytical queries
    {
        "inputs": {"question": "Summarize the key findings from papers about neural architecture search"},
        "metadata": {"category": "analysis", "difficulty": "medium"}
    },
    {
        "inputs": {"question": "Compare different approaches to fine-tuning large language models"},
        "metadata": {"category": "comparison", "difficulty": "hard"}
    },
    
    # Conversational queries
    {
        "inputs": {"question": "Hello, can you help me find research papers?"},
        "metadata": {"category": "greeting", "difficulty": "easy"}
    },
    {
        "inputs": {"question": "Thanks for your help!"},
        "metadata": {"category": "closing", "difficulty": "easy"}
    },
    
    # Edge cases
    {
        "inputs": {"question": "What's the weather today?"},
        "metadata": {"category": "out_of_scope", "difficulty": "easy"}
    },
    {
        "inputs": {"question": ""},
        "metadata": {"category": "empty_input", "difficulty": "easy"}
    },
]

logger.info(f"Evaluation dataset created with {len(eval_data)} test cases")
logger.info("Categories:")
categories = {}
for item in eval_data:
    cat = item["metadata"]["category"]
    categories[cat] = categories.get(cat, 0) + 1

for cat, count in categories.items():
    print(f"  {cat}: {count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create ArxivAgent for Evaluation
# MAGIC
# MAGIC We'll use the real ArxivAgent with MCP tools for evaluation.

# COMMAND ----------

# Create agent with MCP tools
agent = ArxivAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt="You are a helpful research assistant. Use vector search to find papers and answer questions about research.",
    catalog=cfg.catalog,
    schema=cfg.schema,
    genie_space_id=cfg.genie_space_id
)

logger.info("✓ ArxivAgent created for evaluation")
logger.info(f"  - LLM: {cfg.llm_endpoint}")
logger.info(f"  - Vector Search: {cfg.catalog}.{cfg.schema}")
logger.info(f"  - Genie Space: {cfg.genie_space_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define Predict Function

# COMMAND ----------

def predict_fn(question: str) -> str:
    """Wrapper function for agent evaluation.
    
    Args:
        question: User question
        
    Returns:
        Agent response
    """    
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": question}]
    )
    result = agent.predict(request)
    return result.output[-1].content

# Test the predict function
test_response = predict_fn("What are recent papers about transformers?")
logger.info("Test Response:")
logger.info(test_response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Define Domain-Specific Scorers

# COMMAND ----------

# Using pre-defined Guidelines from arxiv_curator.evaluation package
# - polite_tone_guideline: Checks for polite and professional tone
# - scope_guideline: Ensures responses stay focused on research topics
# - hook_in_post_guideline: Checks for engaging introductions
# - word_count_check: Custom scorer for word count limits
# - mentions_papers: Checks if response references research papers

logger.info("Using evaluation scorers from arxiv_curator.evaluation:")
logger.info(f"  ✓ {polite_tone_guideline.name}")
logger.info(f"  ✓ {scope_guideline.name}")
logger.info(f"  ✓ {hook_in_post_guideline.name}")
logger.info(f"  ✓ word_count_check (custom scorer)")
logger.info(f"  ✓ mentions_papers (custom scorer)")

# Judge: Response quality
quality_judge = make_judge(
    name="response_quality",
    instructions=(
        "Evaluate the quality and helpfulness of the response in {{ outputs }} "
        "to the question in {{ inputs }}. Score from 1 to 5:\n"
        "1 - Unhelpful or incorrect\n"
        "2 - Partially helpful but incomplete\n"
        "3 - Adequate and addresses the question\n"
        "4 - Good, clear, and helpful\n"
        "5 - Excellent, comprehensive, and well-explained"
    ),
    model=f"databricks:/{cfg.llm_endpoint}",
    feedback_value_type=int,
)

# Judge: Handles edge cases
edge_case_judge = make_judge(
    name="edge_case_handling",
    instructions=(
        "Evaluate how well the response in {{ outputs }} handles edge cases "
        "like empty inputs, out-of-scope questions, or unclear requests. "
        "Score from 1 to 5:\n"
        "1 - Fails to handle edge case appropriately\n"
        "2 - Attempts to handle but poorly\n"
        "3 - Handles adequately\n"
        "4 - Handles well with clear guidance\n"
        "5 - Excellent handling with helpful redirection"
    ),
    model=f"databricks:/{cfg.llm_endpoint}",
    feedback_value_type=int,
)

# Custom scorer: Word count
@mlflow.genai.scorer
def word_count_check(outputs: list) -> bool:
    """Check that response is between 10 and 350 words."""
    text = outputs[0] if isinstance(outputs[0], str) else outputs[0].get("text", "")
    word_count = len(text.split())
    return 10 <= word_count <= 350

# Custom scorer: Mentions papers
@mlflow.genai.scorer
def mentions_papers(outputs: list) -> bool:
    """Check if response mentions papers or research."""
    text = outputs[0] if isinstance(outputs[0], str) else outputs[0].get("text", "")
    keywords = ["paper", "research", "study", "article", "publication"]
    return any(keyword in text.lower() for keyword in keywords)

logger.info("✓ Scorers defined:")
logger.info("  1. polite_tone_guideline (binary)")
logger.info("  2. scope_guideline (binary)")
logger.info("  3. quality_judge (1-5)")
logger.info("  4. edge_case_judge (1-5)")
logger.info("  5. word_count_check (boolean)")
logger.info("  6. mentions_papers (boolean)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run Evaluation

# COMMAND ----------

# Combine all scorers
all_scorers = [
    polite_tone_guideline,
    scope_guideline,
    quality_judge,
    edge_case_judge,
    word_count_check,
    mentions_papers,
]

# Run evaluation
logger.info("Running evaluation...")
logger.info("=" * 80)

results = mlflow.genai.evaluate(
    predict_fn=predict_fn,
    data=eval_data,
    scorers=all_scorers
)

logger.info("✓ Evaluation complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Analyze Results

# COMMAND ----------

# Display results
logger.info("Evaluation Results:")
logger.info("=" * 80)
display(results.tables['eval_results'])

# COMMAND ----------

# Calculate summary statistics
results_df = results.tables['eval_results']

logger.info("Summary Statistics:")
logger.info("=" * 80)

# Binary metrics
binary_metrics = ["polite_tone/pass", "stays_in_scope/pass", "word_count_check", "mentions_papers"]
for metric in binary_metrics:
    if metric in results_df.columns:
        pass_rate = results_df[metric].mean() * 100
        print(f"{metric}: {pass_rate:.1f}% pass rate")

# Scored metrics
scored_metrics = ["response_quality/score", "edge_case_handling/score"]
for metric in scored_metrics:
    if metric in results_df.columns:
        avg_score = results_df[metric].mean()
        print(f"{metric}: {avg_score:.2f} average")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Analyze by Category

# COMMAND ----------

# Add metadata to results
for i, item in enumerate(eval_data):
    if i < len(results_df):
        results_df.loc[i, 'category'] = item['metadata']['category']
        results_df.loc[i, 'difficulty'] = item['metadata']['difficulty']

# Group by category
logger.info("Results by Category:")
logger.info("=" * 80)

if 'response_quality/score' in results_df.columns and 'category' in results_df.columns:
    category_stats = results_df.groupby('category')['response_quality/score'].agg(['mean', 'count'])
    display(category_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Identify Failure Patterns

# COMMAND ----------

# Find low-scoring cases
if 'response_quality/score' in results_df.columns:
    low_quality = results_df[results_df['response_quality/score'] <= 2]
    
    print(f"\nLow Quality Responses ({len(low_quality)}):")
    print("=" * 80)
    
    for idx, row in low_quality.iterrows():
        print(f"\nQuestion: {eval_data[idx]['inputs']['question']}")
        print(f"Category: {eval_data[idx]['metadata']['category']}")
        print(f"Quality Score: {row['response_quality/score']}")
        print(f"Response: {row['outputs'][:100]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Compare Across Runs

# COMMAND ----------

# Get evaluation history
experiment = mlflow.get_experiment_by_name("/Shared/llmops-course-agent-evaluation")
if experiment:
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=5
    )
    
    print("Recent Evaluation Runs:")
    print("=" * 80)
    
    if len(runs) > 0:
        # Show key metrics over time
        metric_cols = [col for col in runs.columns if 'score' in col or 'pass' in col]
        if metric_cols:
            display(runs[['start_time'] + metric_cols[:5]])
    else:
        print("No previous runs found")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Production Evaluation Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ### Recommended Production Pipeline:
# MAGIC
# MAGIC ```python
# MAGIC # 1. Scheduled Evaluation Job
# MAGIC def run_evaluation_pipeline():
# MAGIC     # Load latest model
# MAGIC     model = load_production_model()
# MAGIC     
# MAGIC     # Load evaluation dataset
# MAGIC     eval_data = load_eval_dataset()
# MAGIC     
# MAGIC     # Run evaluation
# MAGIC     results = mlflow.genai.evaluate(
# MAGIC         predict_fn=model.predict,
# MAGIC         data=eval_data,
# MAGIC         scorers=production_scorers
# MAGIC     )
# MAGIC     
# MAGIC     # Check for regressions
# MAGIC     if results.avg_quality < QUALITY_THRESHOLD:
# MAGIC         send_alert("Quality regression detected!")
# MAGIC     
# MAGIC     # Log results
# MAGIC     log_evaluation_results(results)
# MAGIC     
# MAGIC     return results
# MAGIC
# MAGIC # 2. Run on schedule (e.g., daily)
# MAGIC # Use Databricks Jobs or Workflows
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example: Regression Detection

# COMMAND ----------

def check_for_regression(current_results, threshold=3.5):
    """Check if evaluation results indicate regression.
    
    Args:
        current_results: Current evaluation results
        threshold: Minimum acceptable quality score
        
    Returns:
        bool: True if regression detected
    """
    results_df = current_results.tables['eval_results']
    
    if 'response_quality/score' in results_df.columns:
        avg_quality = results_df['response_quality/score'].mean()
        
        print(f"Average Quality Score: {avg_quality:.2f}")
        print(f"Threshold: {threshold}")
        
        if avg_quality < threshold:
            print(" REGRESSION DETECTED!")
            return True
        else:
            print("✓ Quality meets threshold")
            return False
    
    return False

# Test regression detection
regression_detected = check_for_regression(results, threshold=3.5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Best Practices Summary

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅ Evaluation Best Practices:
# MAGIC
# MAGIC **Dataset Creation:**
# MAGIC 1. Include diverse test cases (easy, medium, hard)
# MAGIC 2. Cover all use case categories
# MAGIC 3. Include edge cases and failure modes
# MAGIC 4. Version your evaluation datasets
# MAGIC 5. Update datasets as product evolves
# MAGIC
# MAGIC **Scorer Selection:**
# MAGIC 1. Use multiple scorers for comprehensive evaluation
# MAGIC 2. Combine guidelines, judges, and custom scorers
# MAGIC 3. Create domain-specific scorers
# MAGIC 4. Validate scorers with human feedback
# MAGIC
# MAGIC **Analysis:**
# MAGIC 1. Analyze results by category
# MAGIC 2. Identify failure patterns
# MAGIC 3. Track metrics over time
# MAGIC 4. Compare across model versions
# MAGIC
# MAGIC **Production:**
# MAGIC 1. Run evaluations on schedule
# MAGIC 2. Set up regression alerts
# MAGIC 3. Log all evaluation results
# MAGIC 4. Review failures regularly
# MAGIC 5. Iterate based on findings

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Register Agent to Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC After evaluation passes, register the agent to Unity Catalog for deployment.

# COMMAND ----------

import subprocess
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path to import root agent.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent import log_register_agent

# Get git SHA (or use a placeholder if not in git repo)
try:
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
except:
    git_sha = "local-dev"

# Define model name in Unity Catalog
model_name = f"{cfg.catalog}.{cfg.schema}.arxiv_agent"

# Calculate evaluation metrics summary
eval_metrics = {}
if 'polite_tone/pass' in results_df.columns:
    eval_metrics['polite_tone_pass_rate'] = results_df['polite_tone/pass'].mean()
if 'word_count_check' in results_df.columns:
    eval_metrics['word_count_pass_rate'] = results_df['word_count_check'].mean()
if 'mentions_papers' in results_df.columns:
    eval_metrics['mentions_papers_rate'] = results_df['mentions_papers'].mean()

logger.info("Registering agent to Unity Catalog...")
logger.info(f"Model name: {model_name}")
logger.info(f"Git SHA: {git_sha}")

# Register the agent (using the agent instance created earlier)
registered_model = log_register_agent(
    agent=agent,
    cfg=cfg,
    git_sha=git_sha,
    run_id=f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    model_name=model_name,
    experiment_path=cfg.experiment_name,
    evaluation_metrics=eval_metrics
)

logger.info(f"✓ Agent registered!")
logger.info(f"  Model: {model_name}")
logger.info(f"  Version: {registered_model.version}")
logger.info(f"  Alias: latest-model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Next Steps

# COMMAND ----------

# MAGIC %md
# MAGIC ### After Registration:
# MAGIC
# MAGIC 1. **Analyze Failures**
# MAGIC    - Why did certain cases fail?
# MAGIC    - Are there patterns?
# MAGIC    - What needs improvement?
# MAGIC
# MAGIC 2. **Improve Agent**
# MAGIC    - Update system prompt
# MAGIC    - Add/improve tools
# MAGIC    - Adjust parameters
# MAGIC    - Fine-tune model (if needed)
# MAGIC
# MAGIC 3. **Re-evaluate**
# MAGIC    - Run evaluation again
# MAGIC    - Compare with baseline
# MAGIC    - Ensure improvements
# MAGIC
# MAGIC 4. **Deploy**
# MAGIC    - If evaluation passes, deploy
# MAGIC    - Monitor in production
# MAGIC    - Continue evaluating