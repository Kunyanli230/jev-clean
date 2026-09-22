"""FakeDecisionClient: deterministic answers injected at the SDK boundary.

Fixtures built on this client are explicitly fake. They exercise the real
detection, candidate, planning, execution, storage and validation code; only the
model boundary is replaced. Fake results never appear in a real clean run.
"""

from __future__ import annotations

from typing import Any

from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer

from idac.models import ClientOutcome, ModelStatus, RequestAttempt, UsageInfo
from idac.settings import RISK_LEVEL_CRITERIA

KEEP = "keep_original"


class FakeDecisionClient:
    """Returns configured distributions for every question of a request."""

    model_identity = "fake"

    def __init__(
        self,
        *,
        approve: bool = True,
        choice_confidence: float = 0.95,
        applicable: float = 0.95,
        risk: dict[int, float] | None = None,
        sdk_score: float | None = None,
        postcheck: float = 0.05,
        overrides: dict[str, dict[str, Any]] | None = None,
        outcomes: list[ClientOutcome] | None = None,
        answers_override: dict[str, Any] | None = None,
        usage: tuple[int | None, int | None] | None = (120, 10),
        request_id_prefix: str = "fake-req",
        budget=None,
    ) -> None:
        self.approve = approve
        self.choice_confidence = choice_confidence
        self.applicable = applicable
        self.risk = risk or {0: 0.90, 1: 0.09, 2: 0.01}
        self.sdk_score = sdk_score
        self.postcheck = postcheck
        self.overrides = overrides or {}
        self.outcomes = list(outcomes or [])
        self.answers_override = answers_override or {}
        self.usage = usage
        self.request_id_prefix = request_id_prefix
        self.budget = budget
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def evaluate(self, state, questions, *, kind: str = "decision") -> ClientOutcome:
        self.calls.append({"kind": kind, "state": state, "questions": questions})
        if self.budget is not None:
            allowed, reason = self.budget.can_attempt()
            if not allowed:
                return ClientOutcome(
                    model_status=ModelStatus.NOT_CALLED, model="fake-jev-1.13.0", error=reason
                )
            self.budget.register_attempt()
        call_number = len(self.calls)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if outcome.model_status == ModelStatus.NOT_CALLED:
                return outcome
            if outcome.model_status != ModelStatus.OK:
                return outcome
        contexts = state.get("candidates", []) if isinstance(state, dict) else []
        answers: dict[str, Any] = {}
        for name in questions:
            if name.endswith("_postcheck"):
                index = int(name[1 : name.index("_")])
                candidate_id = contexts[index]["candidate_id"]
                answers[name] = self.answers_override.get(
                    name, NoulAnswer(noul=self._postcheck_for(candidate_id))
                )
            elif name.endswith("_action"):
                index = int(name[1 : name.index("_")])
                candidate_id = contexts[index]["candidate_id"]
                answers[name] = self.answers_override.get(
                    name, self._choice_answer(candidate_id)
                )
            elif name.endswith("_applicable"):
                index = int(name[1 : name.index("_")])
                candidate_id = contexts[index]["candidate_id"]
                answers[name] = self.answers_override.get(
                    name, NoulAnswer(noul=self._applicable_for(candidate_id))
                )
            elif name.endswith("_risk"):
                index = int(name[1 : name.index("_")])
                candidate_id = contexts[index]["candidate_id"]
                answers[name] = self.answers_override.get(
                    name, self._risk_answer(candidate_id)
                )
            else:  # pragma: no cover - unknown question names are a test bug
                raise AssertionError(f"unexpected question {name}")
        input_tokens, output_tokens = self.usage if self.usage else (None, None)
        if self.budget is not None:
            self.budget.register_usage(input_tokens, output_tokens)
        return ClientOutcome(
            model_status=ModelStatus.OK,
            model="fake-jev-1.13.0",
            request_id=f"{self.request_id_prefix}-{call_number:04d}",
            answers=answers,
            usage=UsageInfo(
                input_tokens=input_tokens, output_tokens=output_tokens, known=input_tokens is not None
            ),
            attempts=[RequestAttempt(attempt=1, status="ok")],
            elapsed_seconds=0.001,
        )

    def _config(self, candidate_id: str) -> dict[str, Any]:
        return self.overrides.get(candidate_id, {})

    def _choice_answer(self, candidate_id: str) -> ChoiceAnswer:
        config = self._config(candidate_id)
        choice = config.get("choice", candidate_id if self.approve else KEEP)
        confidence = config.get("choice_confidence", self.choice_confidence)
        if choice == candidate_id:
            probabilities = {candidate_id: confidence, KEEP: 1.0 - confidence}
        else:
            probabilities = {candidate_id: 1.0 - confidence, KEEP: confidence}
        return ChoiceAnswer(
            choice=choice, confidence=confidence, probabilities=probabilities
        )

    def _applicable_for(self, candidate_id: str) -> float:
        return self._config(candidate_id).get("applicable", self.applicable)

    def _postcheck_for(self, candidate_id: str) -> float:
        return self._config(candidate_id).get("postcheck", self.postcheck)

    def _risk_answer(self, candidate_id: str) -> ScoreAnswer:
        config = self._config(candidate_id)
        distribution = config.get("risk", self.risk)
        score = config.get("sdk_score", self.sdk_score)
        if score is None:
            score = sum(level * probability for level, probability in distribution.items())
        return ScoreAnswer(
            score=score,
            confidence=config.get("risk_confidence", 0.9),
            legend=dict(enumerate(RISK_LEVEL_CRITERIA)),
            probabilities=dict(distribution),
        )


def failing_outcome(status: ModelStatus, error: str) -> ClientOutcome:
    return ClientOutcome(model_status=status, model="fake-jev-1.13.0", error=error)


# ---------------------------------------------------------------------------
# Shared fixture helpers (kept here so the fixed test file list stays complete)
# ---------------------------------------------------------------------------

import csv  # noqa: E402
from pathlib import Path  # noqa: E402

DEMO_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "demo.yaml"
DEMO_DATA = Path(__file__).resolve().parent.parent / "examples" / "data"


def write_config(tmp_path: Path, text: str, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def write_csv_file(
    tmp_path: Path, columns: list[str], rows: list[list[str]], name: str = "input.csv"
) -> Path:
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)
    return path


def run_fake_pipeline(
    tmp_path: Path,
    config_text: str,
    columns: list[str],
    rows: list[list[str]],
    client: FakeDecisionClient | None = None,
    *,
    output_name: str = "run",
    budget=None,
):
    from idac.orchestrator import run_clean
    from idac.storage import RunStore

    config_path = write_config(tmp_path, config_text)
    input_path = write_csv_file(tmp_path, columns, rows)
    client = client or FakeDecisionClient()
    output = tmp_path / output_name
    result = run_clean(input_path, config_path, output, client=client, budget=budget)
    return result, RunStore(output), client
