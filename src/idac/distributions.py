"""Explicit distribution validation and statistics.

All statistics are pure functions over the SDK's typed answers. Invalid
responses are rejected, never silently zero-filled, clipped or renormalized.
The expected risk level is always recomputed from the full distribution so a
decision can be replayed offline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from .models import ChoiceAssessment, ReasonCode, RiskAssessment
from .settings import PROBABILITY_SUM_TOLERANCE, RISK_LEVEL_CRITERIA


class InvalidDistributionError(ValueError):
    """A model answer that cannot be used for a decision."""

    def __init__(self, reason: ReasonCode, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _validate_probability_map(
    probabilities: Mapping[object, float],
    expected_keys: set[object],
    label: str,
) -> dict[object, float]:
    if set(probabilities) != expected_keys:
        raise InvalidDistributionError(
            ReasonCode.INVALID_DISTRIBUTION,
            f"{label} keys {sorted(map(str, probabilities))} do not match expected "
            f"{sorted(map(str, expected_keys))}",
        )
    clean: dict[object, float] = {}
    for key, value in probabilities.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise InvalidDistributionError(
                ReasonCode.INVALID_DISTRIBUTION, f"{label}[{key}] is not numeric: {value!r}"
            )
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise InvalidDistributionError(
                ReasonCode.INVALID_DISTRIBUTION, f"{label}[{key}] is outside [0, 1]: {value!r}"
            )
        clean[key] = number
    total = sum(clean.values())
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise InvalidDistributionError(
            ReasonCode.INVALID_DISTRIBUTION,
            f"{label} probabilities sum to {total!r}, outside tolerance "
            f"{PROBABILITY_SUM_TOLERANCE!r}",
        )
    return clean


def summarize_choice(
    answer,
    option_descriptions: dict[str, str],
    candidate_id: str,
) -> ChoiceAssessment:
    """Validate a Choice answer and record its full distribution."""
    probabilities = _validate_probability_map(
        dict(answer.probabilities), set(option_descriptions), "choice"
    )
    selected = answer.choice
    if selected not in probabilities:
        raise InvalidDistributionError(
            ReasonCode.INVALID_DISTRIBUTION,
            f"selected choice {selected!r} is not one of the offered options",
        )
    confidence = answer.confidence
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not math.isfinite(float(confidence))
        or not (0.0 <= float(confidence) <= 1.0)
    ):
        raise InvalidDistributionError(
            ReasonCode.INVALID_DISTRIBUTION, f"choice confidence is invalid: {confidence!r}"
        )
    ordered = sorted(probabilities.values(), reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    return ChoiceAssessment(
        selected_option=selected,
        option_descriptions=dict(option_descriptions),
        probabilities={str(key): value for key, value in probabilities.items()},
        selected_probability=float(probabilities[selected]),
        top_two_margin=float(margin),
        sdk_confidence=float(confidence),
    )


def expected_level(probabilities: Mapping[int, float]) -> float:
    """mu = sum(k * P(k)) over the project's ordered 0/1/2 encoding."""
    return sum(float(level) * float(probability) for level, probability in probabilities.items())


def level_variance(probabilities: Mapping[int, float], mean: float) -> float:
    """Var(L) = sum(P(k) * (k - mu)^2)."""
    return sum(
        float(probability) * (float(level) - mean) ** 2
        for level, probability in probabilities.items()
    )


def high_risk_probability(probabilities: Mapping[int, float]) -> float:
    return float(probabilities.get(2, 0.0))


def summarize_risk(answer, expected_legend: dict[int, str] | None = None) -> RiskAssessment:
    """Validate a Score answer and record the full ordered distribution."""
    legend_raw = {int(level): str(text) for level, text in answer.legend.items()}
    expected_legend = (
        dict(enumerate(RISK_LEVEL_CRITERIA)) if expected_legend is None else expected_legend
    )
    if legend_raw != expected_legend or set(legend_raw) != {0, 1, 2}:
        raise InvalidDistributionError(
            ReasonCode.INVALID_DISTRIBUTION,
            "score legend does not match the requested ordered risk criteria",
        )
    probabilities = _validate_probability_map(
        {int(level): value for level, value in answer.probabilities.items()},
        {0, 1, 2},
        "score",
    )
    score = answer.score
    confidence = answer.confidence
    for name, value in (("score", score), ("score confidence", confidence)):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not (0.0 <= float(value) <= 2.0 if name == "score" else 0.0 <= float(value) <= 1.0)
        ):
            raise InvalidDistributionError(
                ReasonCode.INVALID_DISTRIBUTION, f"{name} is invalid: {value!r}"
            )
    probabilities_int = {int(level): value for level, value in probabilities.items()}
    mean = expected_level(probabilities_int)
    return RiskAssessment(
        legend=legend_raw,
        probabilities=probabilities_int,
        sdk_score=float(score),
        expected_risk_level=mean,
        score_expectation_delta=float(score) - mean,
        variance=level_variance(probabilities_int, mean),
        high_risk_probability=high_risk_probability(probabilities_int),
        sdk_confidence=float(confidence),
    )


def validate_noul(answer) -> float:
    """Validate a Noul answer; P(true) must be finite and inside [0, 1]."""
    value = answer.noul
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not (0.0 <= float(value) <= 1.0)
    ):
        raise InvalidDistributionError(
            ReasonCode.INVALID_DISTRIBUTION, f"noul probability is invalid: {value!r}"
        )
    return float(value)


def expected_expansion(probabilities: Mapping[int, float]) -> str:
    """Human-readable expansion of mu with the actual probabilities."""
    terms = " + ".join(
        f"{level}*{float(probability):.6g}" for level, probability in sorted(probabilities.items())
    )
    return f"{terms} = {expected_level(probabilities):.6g}"
