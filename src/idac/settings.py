"""Fixed model, budget and policy constants for IDAC.

Every threshold in this module is a course-project policy choice, not an official
accuracy guarantee from TypeSafe. The values are frozen here so that decision
records can snapshot exactly which policy produced a decision.
"""

from __future__ import annotations

MODEL_NAME = "jev-1.13.0"
BASE_URL = "https://api.typesafe.ai"
API_KEY_ENV = "TYPESAFE_API_KEY"

REQUEST_TIMEOUT_SECONDS = 45.0
RUN_TIME_LIMIT_SECONDS = 600.0
MAX_SDK_REQUEST_ATTEMPTS = 120
MAX_INPUT_TOKENS = 200_000

RETRY_MAX_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 2.0

MAX_CANDIDATES_PER_BATCH = 4
MAX_SAMPLES_PER_CANDIDATE = 8
MAX_CELL_CHARS = 200
MAX_REQUEST_BYTES = 24 * 1024

MIN_CHOICE_CONFIDENCE = 0.80
MIN_APPLICABLE_PROBABILITY = 0.80
MAX_SEMANTIC_RISK_SCORE = 0.50
MAX_HIGH_RISK_PROBABILITY = 0.10
MAX_POSTCHECK_ERROR_PROBABILITY = 0.20
SCORE_EXPECTATION_TOLERANCE = 0.03
PROBABILITY_SUM_TOLERANCE = 1e-6

POLICY_VERSION = "idac-policy-1"
QUESTION_VERSION = "idac-questions-1"

RISK_LEVEL_CRITERIA = [
    "Low: representation-only change or explicitly permitted representative-value handling.",
    "Moderate: insufficient meaning or a plausible competing interpretation.",
    "High: likely distortion or loss of meaningful information.",
]

MAX_ROWS = 10_000
MAX_COLUMNS = 30

MIN_IQR_SAMPLE = 4
IQR_MULTIPLIER = 1.5


def policy_snapshot() -> dict[str, float | int | str]:
    """Return the exact policy values stored with every decision record."""
    return {
        "policy_version": POLICY_VERSION,
        "question_version": QUESTION_VERSION,
        "model": MODEL_NAME,
        "min_choice_confidence": MIN_CHOICE_CONFIDENCE,
        "min_applicable_probability": MIN_APPLICABLE_PROBABILITY,
        "max_semantic_risk_score": MAX_SEMANTIC_RISK_SCORE,
        "max_high_risk_probability": MAX_HIGH_RISK_PROBABILITY,
        "max_postcheck_error_probability": MAX_POSTCHECK_ERROR_PROBABILITY,
        "score_expectation_tolerance": SCORE_EXPECTATION_TOLERANCE,
        "probability_sum_tolerance": PROBABILITY_SUM_TOLERANCE,
    }
