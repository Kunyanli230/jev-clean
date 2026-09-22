"""Decision cards, distribution tables and expected-level expansions.

Everything rendered here comes from stored records; no value is pre-filled from
a template and no model call is made. Display rounding is marked as such and the
JSON references keep full precision for replay.
"""

from __future__ import annotations

from typing import Any

from .distributions import expected_expansion, expected_level
from .models import CheckStatus, DecisionRecord, DecisionTrace
from .settings import MAX_POSTCHECK_ERROR_PROBABILITY

DISPLAY_NOTE = "Values are rounded for display; the stored JSON keeps full precision."


def _as_record(record: DecisionRecord | dict[str, Any]) -> DecisionRecord:
    if isinstance(record, dict):
        return DecisionRecord.model_validate(record)
    return record


def decision_card(
    record: DecisionRecord | dict[str, Any],
    trace: DecisionTrace | dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = _as_record(record)
    if isinstance(trace, dict):
        trace = DecisionTrace.model_validate(trace)
    card: dict[str, Any] = {
        "decision_id": record.id,
        "candidate_id": record.candidate_id,
        "role": record.role,
        "object": {
            "operation": record.operation,
            "column": record.column,
            "target_count": record.target_count,
            "version_id": record.version_id,
            "evidence_ids": record.evidence_ids,
            "context_ref": record.context_ref,
            "questions_ref": record.questions_ref,
        },
        "model": {
            "model": record.model,
            "model_status": record.model_status.value,
            "request_id": record.request_id,
            "question_version": record.question_version,
            "policy_version": record.policy_version,
        },
        "options": None,
        "choice": None,
        "applicable": None,
        "risk": None,
        "expected_expansion": None,
        "rules": [
            {
                "rule_id": check.rule_id,
                "metric": check.metric,
                "observed_value": check.observed_value,
                "operator": check.operator,
                "threshold_or_required_value": check.threshold_or_required_value,
                "status": check.status.value,
                "reason_code": check.reason_code.value if check.reason_code else None,
            }
            for check in record.gate_checks
        ],
        "decision": {
            "gate_accepted": record.gate_accepted,
            "reason_codes": [reason.value for reason in record.reason_codes],
        },
        "execution": None,
    }
    if record.choice_assessment is not None:
        choice = record.choice_assessment
        card["options"] = [
            {
                "option": option,
                "description": description,
                "probability": choice.probabilities.get(option),
                "selected": option == choice.selected_option,
            }
            for option, description in choice.option_descriptions.items()
        ]
        card["choice"] = {
            "selected_option": choice.selected_option,
            "selected_probability": choice.selected_probability,
            "other_probability": 1.0 - choice.selected_probability,
            "top_two_margin": choice.top_two_margin,
            "sdk_confidence": choice.sdk_confidence,
            "distribution": choice.probabilities,
        }
    if record.applicable_probability is not None:
        card["applicable"] = {
            "p_true": record.applicable_probability,
            "p_false": 1.0 - record.applicable_probability,
        }
    if record.risk_assessment is not None:
        risk = record.risk_assessment
        card["risk"] = {
            "legend": risk.legend,
            "distribution": risk.probabilities,
            "sdk_score": risk.sdk_score,
            "expected_risk_level": risk.expected_risk_level,
            "score_expectation_delta": risk.score_expectation_delta,
            "variance": risk.variance,
            "high_risk_probability": risk.high_risk_probability,
            "sdk_confidence": risk.sdk_confidence,
        }
        card["expected_expansion"] = expected_expansion(risk.probabilities)
    card["execution"] = {
        "planner_status": trace.planner_status if trace else "not_planned",
        "hard_validation_status": trace.hard_validation_status if trace else "not_run",
        "postcheck_status": trace.postcheck_status if trace else "not_run",
        "postcheck_probability": trace.postcheck_probability if trace else None,
        "postcheck_ok_probability": (
            1.0 - trace.postcheck_probability
            if trace and trace.postcheck_probability is not None
            else None
        ),
        "postcheck_threshold": MAX_POSTCHECK_ERROR_PROBABILITY,
        "commit_status": trace.commit_status if trace else "not_committed",
        "committed_version": trace.committed_version if trace else None,
        "blocking_candidate_ids": trace.blocking_candidate_ids if trace else [],
        "trace_reason_codes": [reason.value for reason in trace.reason_codes] if trace else [],
    }
    return card


def render_distribution_table(distribution: dict[Any, float], legend: dict[Any, str] | None = None) -> str:
    lines = ["| Level | Meaning | Probability |", "|---|---|---:|"]
    for level in sorted(distribution):
        meaning = (legend or {}).get(level, "")
        lines.append(f"| {level} | {meaning} | {distribution[level]:.6f} |")
    return "\n".join(lines)


def render_options_table(options: list[dict[str, Any]]) -> str:
    lines = ["| Option | Description | Probability | Selected |", "|---|---|---:|:---:|"]
    for option in options:
        probability = option["probability"]
        rendered = "not obtained" if probability is None else f"{probability:.6f}"
        lines.append(
            f"| {option['option']} | {option['description']} | {rendered} | "
            f"{'yes' if option['selected'] else 'no'} |"
        )
    return "\n".join(lines)


def render_rule_table(rules: list[dict[str, Any]]) -> str:
    lines = [
        "| Rule | Metric | Observed | Operator | Threshold | Status | Reason code |",
        "|---|---|---|:---:|---|:---:|---|",
    ]
    for rule in rules:
        observed = rule["observed_value"]
        rendered = "not obtained" if observed is None else str(observed)
        lines.append(
            f"| {rule['rule_id']} | {rule['metric']} | {rendered} | {rule['operator']} | "
            f"{rule['threshold_or_required_value']} | {rule['status']} | "
            f"{rule['reason_code'] or ''} |"
        )
    return "\n".join(lines)


def render_card_markdown(card: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"### Decision card `{card['decision_id']}`")
    lines.append("")
    lines.append("**Object and evidence**")
    lines.append("")
    obj = card["object"]
    lines.append(f"- candidate: `{card['candidate_id']}` (role `{card['role']}`)")
    lines.append(
        f"- operation: `{obj['operation']}` on column `{obj['column']}` "
        f"covering {obj['target_count']} target rows"
    )
    lines.append(f"- snapshot: `{obj['version_id']}`")
    lines.append(f"- evidence ids: {', '.join(obj['evidence_ids']) or 'none'}")
    lines.append(f"- context reference: `{obj['context_ref']}`")
    lines.append(f"- questions reference: `{obj['questions_ref']}`")
    lines.append("")
    model = card["model"]
    lines.append(
        f"- model: `{model['model']}`; model_status `{model['model_status']}`; "
        f"request_id `{model['request_id']}`"
    )
    lines.append(
        f"- question_version `{model['question_version']}`; policy_version `{model['policy_version']}`"
    )
    lines.append("")
    if card["options"]:
        lines.append("**Available options**")
        lines.append("")
        lines.append(render_options_table(card["options"]))
        lines.append("")
    if card["choice"]:
        choice = card["choice"]
        lines.append("**Choice distribution**")
        lines.append("")
        lines.append(
            f"selected `{choice['selected_option']}`; P(selected)={choice['selected_probability']:.6f}; "
            f"P(other)={choice['other_probability']:.6f}; top-two margin={choice['top_two_margin']:.6f}; "
            f"SDK confidence={choice['sdk_confidence']:.6f}"
        )
        lines.append("")
        for option, probability in choice["distribution"].items():
            lines.append(f"- `{option}`: {probability:.6f}")
        lines.append("")
    if card["applicable"]:
        lines.append("**Applicable (Noul)**")
        lines.append("")
        lines.append(
            f"P(true)={card['applicable']['p_true']:.6f}; P(false)={card['applicable']['p_false']:.6f}"
        )
        lines.append("")
    if card["risk"]:
        risk = card["risk"]
        lines.append("**Risk distribution (Score)**")
        lines.append("")
        lines.append(render_distribution_table(risk["distribution"], risk["legend"]))
        lines.append("")
        lines.append(
            f"SDK score={risk['sdk_score']:.6f}; SDK confidence={risk['sdk_confidence']:.6f}; "
            f"expected level={risk['expected_risk_level']:.6f}; "
            f"delta={risk['score_expectation_delta']:.6f}; variance={risk['variance']:.6f}; "
            f"P(L=2)={risk['high_risk_probability']:.6f}"
        )
        lines.append("")
        lines.append(f"Expected level expansion: `{card['expected_expansion']}`")
        lines.append("")
    lines.append("**Rule table**")
    lines.append("")
    lines.append(render_rule_table(card["rules"]))
    lines.append("")
    execution = card["execution"]
    decision = card["decision"]
    lines.append("**Decision and execution status**")
    lines.append("")
    lines.append(
        f"- gate accepted: {decision['gate_accepted']}; reason codes: "
        f"{', '.join(decision['reason_codes']) or 'none'}"
    )
    postcheck_p = execution["postcheck_probability"]
    postcheck_display = (
        "not obtained"
        if postcheck_p is None
        else (
            f"P(semantic error)={postcheck_p}, P(no semantic error)="
            f"{execution['postcheck_ok_probability']}, threshold={execution['postcheck_threshold']}"
        )
    )
    lines.append(
        f"- planner: {execution['planner_status']}; hard validation: "
        f"{execution['hard_validation_status']}; postcheck: {execution['postcheck_status']} "
        f"({postcheck_display})"
    )
    lines.append(
        f"- commit: {execution['commit_status']}"
        + (
            f" at `{execution['committed_version']}`"
            if execution["committed_version"]
            else ""
        )
    )
    if execution["blocking_candidate_ids"]:
        lines.append(
            f"- batch rejected; triggering candidates: "
            f"{', '.join(execution['blocking_candidate_ids'])}"
        )
    lines.append("")
    lines.append(f"_{DISPLAY_NOTE}_")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def render_decision_table(
    records: list[DecisionRecord | dict[str, Any]],
    traces: list[DecisionTrace | dict[str, Any]] | None = None,
) -> str:
    trace_map = {}
    for trace in traces or []:
        if isinstance(trace, dict):
            trace = DecisionTrace.model_validate(trace)
        trace_map[trace.decision_id] = trace
    lines = [
        "| Candidate | Operation | Column | P(apply) | P(keep) | Choice conf. | P(applicable) | "
        "P(L=0) | P(L=1) | P(L=2) | Expected | P(high risk) | Gate | Failed rules | Commit |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|---|",
    ]
    for raw in records:
        record = _as_record(raw)
        trace = trace_map.get(record.id)
        choice = record.choice_assessment
        risk = record.risk_assessment
        distribution = risk.probabilities if risk else {}
        failed = [
            check.rule_id
            for check in record.gate_checks
            if check.status == CheckStatus.FAILED
        ]
        commit = trace.commit_status if trace else "not_committed"
        lines.append(
            "| `{candidate}` | {operation} | {column} | {apply} | {keep} | {confidence} | "
            "{applicable} | {l0} | {l1} | {l2} | {expected} | {high} | {gate} | {failed} | {commit} |".format(
                candidate=record.candidate_id,
                operation=record.operation or "",
                column=record.column or "",
                apply=_fmt(choice.selected_probability if choice else None),
                keep=_fmt(1.0 - choice.selected_probability if choice else None),
                confidence=_fmt(choice.sdk_confidence if choice else None),
                applicable=_fmt(record.applicable_probability),
                l0=_fmt(distribution.get(0)),
                l1=_fmt(distribution.get(1)),
                l2=_fmt(distribution.get(2)),
                expected=_fmt(risk.expected_risk_level if risk else None),
                high=_fmt(risk.high_risk_probability if risk else None),
                gate="accepted" if record.gate_accepted else "abstained",
                failed=", ".join(failed) or "none",
                commit=commit,
            )
        )
    lines.append("")
    lines.append(f"_{DISPLAY_NOTE}_")
    return "\n".join(lines)


def render_expected_examples() -> str:
    """The fixed synthetic A/B illustration; never presented as real model output."""
    rows = [
        ("A", {0: 0.60, 1: 0.40, 2: 0.00}),
        ("B", {0: 0.80, 1: 0.00, 2: 0.20}),
    ]
    lines = [
        "### Synthetic illustration: same expected level, different tail risk",
        "",
        "| Example | P(0) | P(1) | P(2) | Expected level | Variance | Expected-level gate | High-risk gate |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for name, distribution in rows:
        mean = expected_level(distribution)
        variance = sum(p * (level - mean) ** 2 for level, p in distribution.items())
        expected_gate = "pass" if mean <= 0.50 else "fail"
        high_gate = "pass" if distribution[2] <= 0.10 else "fail"
        lines.append(
            f"| {name} | {distribution[0]:.2f} | {distribution[1]:.2f} | {distribution[2]:.2f} | "
            f"{mean:.2f} | {variance:.2f} | {expected_gate} | {high_gate} |"
        )
    lines.append("")
    lines.append(
        "These are **synthetic illustrations** of the project's ordered 0/1/2 encoding, not "
        "real Jev results. Both examples share expected level 0.40, but B has P(L=2)=0.20 and "
        "must abstain because the high-risk probability exceeds 0.10. The expected level alone "
        "cannot explain the difference."
    )
    return "\n".join(lines)
