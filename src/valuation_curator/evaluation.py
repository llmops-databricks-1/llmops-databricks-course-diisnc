import re

import mlflow
from mlflow.genai.scorers import Guidelines

from valuation_curator.agent import ValuationAgent
from valuation_curator.config import ProjectConfig

# these are guidelines, not judges
# polite_tone_guideline = Guidelines(
#     name="polite_tone",
#     guidelines=[
#         "The response must use a polite and professional tone throughout",
#         "The response should be friendly and helpful without being condescending",
#         "The response must avoid any dismissive or rude language",
#     ],
#     model="databricks:/databricks-gpt-oss-120b",
# )

scope_guideline = Guidelines(
    name="stays_in_scope",
    guidelines=[
        "The response must only discuss topics related to customs valuation documents",
        "The response should not answer questions about unrelated topics",
        "If asked about non-customs valuation topics, politely redirect to customs "
        "valuation-related questions",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)

# this is an example of a bad guideline (subjective)
hook_in_post_guideline = Guidelines(
    name="hook_in_post",
    guidelines=[
        "The response must start with an engaging hook that captures attention",
        "The opening should make the reader want to continue reading",
        "The response should have a compelling introduction before diving into details",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)

actionable_recommendations_guideline = Guidelines(
    name="actionable_recommendations",
    guidelines=[
        "The response must include concrete next steps for investigation",
        "Recommendations should be specific, testable, and operationally feasible",
        "The response should avoid vague advice such as 'look into this'",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)

professional_audit_tone_guideline = Guidelines(
    name="professional_audit_tone",
    guidelines=[
        "The response must use a concise, professional, audit-ready tone",
        "The response should prioritize facts, evidence, and traceable statements",
        "The response must avoid casual or social-media style language",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)

no_unwarranted_conclusions_guideline = Guidelines(
    name="no_unwarranted_conclusions",
    guidelines=[
        "The response must not claim fraud or legal violations "
        "without clear supporting evidence",
        "The response should use cautious language when evidence is incomplete",
        "The response should distinguish observed anomalies from confirmed wrongdoing",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)


def evaluate_agent(cfg: ProjectConfig, eval_inputs_path: str) -> mlflow.models.EvaluationResult:
    """Run evaluation on the agent.

    Args:
        cfg: Project configuration.
        eval_inputs_path: Path to evaluation inputs file.

    Returns:
        MLflow EvaluationResult with metrics.
    """
    agent = ValuationAgent(
        llm_endpoint=cfg.llm_endpoint,
        system_prompt=cfg.system_prompt,
        catalog=cfg.catalog,
        schema=cfg.schema,
        genie_space_id=cfg.genie_space_id,
        lakebase_project_id=cfg.lakebase_project_id,
    )

    with open(eval_inputs_path) as f:
        eval_data = [{"inputs": {"question": line.strip()}} for line in f if line.strip()]

    def predict_fn(question: str) -> str:
        request = {"input": [{"role": "user", "content": question}]}
        result = agent.predict(request)
        return result.output[-1].content

    return mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=eval_data,
        scorers=[
            word_count_check,
            mentions_valuation_docs,
            evidence_citations_scorer,
            uses_detect_anomalies_tool,
            # polite_tone_guideline,
            scope_guideline,
            hook_in_post_guideline,
            actionable_recommendations_guideline,
            professional_audit_tone_guideline,
            no_unwarranted_conclusions_guideline,
        ],
    )


@mlflow.genai.scorer
def word_count_check(outputs: list) -> bool:
    """Check that the output is under 350 words.

    Args:
        outputs: List of output dictionaries

    Returns:
        True if word count is under 350, False otherwise
    """
    # Handle different output formats
    if isinstance(outputs, list) and len(outputs) > 0:
        if isinstance(outputs[0], dict) and "text" in outputs[0]:
            text = outputs[0]["text"]
        elif isinstance(outputs[0], str):
            text = outputs[0]
        else:
            text = str(outputs[0])
    else:
        text = str(outputs)

    word_count = len(text.split())
    return word_count < 350


@mlflow.genai.scorer
def mentions_valuation_docs(outputs: list) -> bool:
    """Check if the response mentions specific customs valuation documents.

    Args:
        outputs: List of output dictionaries

    Returns:
        True if customs valuation documents are mentioned, False otherwise
    """
    # Handle different output formats
    if isinstance(outputs, list) and len(outputs) > 0:
        if isinstance(outputs[0], dict) and "text" in outputs[0]:
            text = outputs[0]["text"]
        elif isinstance(outputs[0], str):
            text = outputs[0]
        else:
            text = str(outputs[0])
    else:
        text = str(outputs)

    text_lower = text.lower()
    keywords = [
        "invoice",
        "royalty",
        "declaration",
        "valuation",
        "invoice total",
        "vat",
        "currency",
    ]

    return any(keyword in text_lower for keyword in keywords)


@mlflow.genai.scorer
def evidence_citations_scorer(outputs: list) -> int:
    """Score how many evidence signals are present in the response (0 – 5).

    Checks 5 signal types:
      1. Field keywords (case_id, invoice total, anomaly, etc.)
      2. Exact anomaly labels (invoice_total_mismatch, etc.)
      3. Case ID reference pattern
      4. Numeric values (amounts, percentages)
      5. Source/document mention

    Returns the number of signals found (0 if none, up to 5).
    """
    if isinstance(outputs, list) and len(outputs) > 0:
        if isinstance(outputs[0], dict) and "text" in outputs[0]:
            text = outputs[0]["text"]
        elif isinstance(outputs[0], str):
            text = outputs[0]
        else:
            text = str(outputs[0])
    else:
        text = str(outputs)

    text_lower = text.lower()
    signals = 0

    field_keywords = [
        "case_id",
        "invoice total",
        "declaration total",
        "declaration invoice total",
        "royalty percentage",
        "declaration royalty",
        "anomaly",
        "mismatch",
    ]
    if any(keyword in text_lower for keyword in field_keywords):
        signals += 1

    anomaly_labels = [
        "invoice_total_mismatch",
        "declaration_total_mismatch",
        "royalty_mismatch",
    ]
    if any(label in text_lower for label in anomaly_labels):
        signals += 1

    if re.search(r"\bcase[-_ ]?id\b|\bcase\s*[=:]\s*[a-z0-9_-]+", text_lower):
        signals += 1

    if re.search(r"\b\d+(?:\.\d+)?%\b|\b\$?\d+(?:,\d{3})*(?:\.\d+)?\b", text):
        signals += 1

    if "source" in text_lower or "document" in text_lower:
        signals += 1

    return signals


@mlflow.genai.scorer
def uses_detect_anomalies_tool(outputs: list) -> bool:
    """Proxy check that the response reflects detect_anomalies tool usage.

    Since scorer inputs don't include trace spans, this checks for evidence patterns
    that typically result from detect_anomalies outputs.
    """
    if isinstance(outputs, list) and len(outputs) > 0:
        if isinstance(outputs[0], dict) and "text" in outputs[0]:
            text = outputs[0]["text"]
        elif isinstance(outputs[0], str):
            text = outputs[0]
        else:
            text = str(outputs[0])
    else:
        text = str(outputs)

    text_lower = text.lower()
    patterns = [
        "invoice_total_mismatch",
        "declaration_total_mismatch",
        "royalty_mismatch",
        "anomaly",
        "mismatch",
    ]
    return any(pattern in text_lower for pattern in patterns)


def create_eval_data_from_file(eval_inputs_path: str) -> list[dict]:
    """Load evaluation data from a file.

    Args:
        eval_inputs_path: Path to file with one question per line

    Returns:
        List of evaluation data dictionaries
    """
    with open(eval_inputs_path) as f:
        eval_data = [{"inputs": {"question": line.strip()}} for line in f if line.strip()]
    return eval_data
