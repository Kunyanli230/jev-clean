"""Deterministic orchestrator: the fixed seven sub-phases and transaction loop.

Each sub-phase re-checks the current snapshot, builds fully computed candidates,
routes them to the role agent, evaluates the decision gate, plans, executes,
hard-validates, semantically verifies and then commits or rejects. No model
request is sent for an empty phase and no rejected candidate is re-sampled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agents import AgentBatchResult, agent_for_role
from .candidates import build_candidates
from .config import TableConfig, load_config
from .executor import ExecutionError, execute
from .models import (
    PHASE_ORDER,
    ClientOutcome,
    DatasetState,
    DecisionRecord,
    DecisionTrace,
    Evidence,
    IssueStatus,
    ModelStatus,
    Phase,
    ReasonCode,
    RepairCandidate,
    RepairPlan,
    RunOutcome,
    ValidationResult,
    fingerprint,
)
from .planner import AcceptedCandidate, build_plan
from .profiling import PhaseDetection, detect_phase
from .questions import (
    OPERATION_ROLE,
    build_candidate_context,
    build_postcheck_contexts,
    build_postcheck_questions,
    build_questions,
    serialize_questions,
)
from .settings import (
    MAX_CANDIDATES_PER_BATCH,
    MAX_REQUEST_BYTES,
    MAX_SAMPLES_PER_CANDIDATE,
    MODEL_NAME,
    QUESTION_VERSION,
)
from .storage import RunStore, load_csv, mark_outcome
from .typesafe_client import ConfigurationError, DecisionClient, RunBudget, TypeSafeDecisionClient
from .validator import confirmed_repaired, validate_hard, validate_semantic


@dataclass
class CleanResult:
    run_dir: Path
    outcome: RunOutcome
    stop_reason: str
    current_version: str
    versions: list[str] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    model_identity: str = "real"
    configuration_error: bool = False


class _RunContext:
    def __init__(
        self,
        store: RunStore,
        config: TableConfig,
        state: DatasetState,
        client: DecisionClient,
        budget: RunBudget,
    ) -> None:
        self.store = store
        self.config = config
        self.state = state
        self.client = client
        self.budget = budget
        self.evidence: dict[str, Evidence] = {}
        self.validation_results: list[ValidationResult] = []
        self.records: list[DecisionRecord] = []
        self.traces: list[DecisionTrace] = []
        self.version_counter = 0
        self.plan_counter = 0
        self.request_counter = 0
        self.stop_reason: str | None = None
        self.budget_stop = False

    def next_version(self) -> str:
        self.version_counter += 1
        return f"v{self.version_counter:03d}"

    def next_plan(self) -> str:
        self.plan_counter += 1
        return f"plan-{self.plan_counter:04d}"

    def next_request(self) -> int:
        self.request_counter += 1
        return self.request_counter


def run_clean(
    input_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    client: DecisionClient | None = None,
    api_key: str | None = None,
    budget: RunBudget | None = None,
) -> CleanResult:
    """Execute the fixed cleaning pipeline and write the run directory."""
    config = load_config(config_path)
    owned_client = client is None
    budget = budget or RunBudget()
    if client is None:
        client = TypeSafeDecisionClient(budget=budget, api_key=api_key)
    store = RunStore.create(
        output_dir, config_path, input_path, config, model_identity=client.model_identity
    )
    state = load_csv(input_path, config)
    store.save_snapshot(state)
    store.record_version(state, plan_id=None, candidate_ids=[], summary={"kind": "original"})
    store.update_manifest(state="LOADED")
    store.audit("run_created", version_id=state.version_id, input_hash=state.input_hash)
    context = _RunContext(store, config, state, client, budget)
    try:
        context.state = _run_phases(context)
        outcome = _finalize(context)
    except ConfigurationError as error:
        _finish_failed(context, ReasonCode.CONFIGURATION_FAILURE.value)
        if owned_client:
            client.close()
        raise ConfigurationError(str(error)) from error
    except Exception as error:  # pragma: no cover - defensive finalization
        _finish_failed(context, f"UNEXPECTED_ERROR: {type(error).__name__}")
        if owned_client:
            client.close()
        raise
    if owned_client:
        client.close()
    return CleanResult(
        run_dir=Path(output_dir),
        outcome=outcome,
        stop_reason=context.stop_reason or "no_actionable_issues",
        current_version=context.store.manifest()["current_version"],
        versions=context.store.ancestor_chain(),
        budget=budget.snapshot(),
        model_identity=client.model_identity,
    )


def _finish_failed(context: _RunContext, stop_reason: str) -> None:
    context.stop_reason = stop_reason
    try:
        context.store.save_snapshot(context.state)
        context.store.export_current()
        mark_outcome(context.store, RunOutcome.FAILED, stop_reason, "FAILED")
        context.store.audit("run_failed", details={"stop_reason": stop_reason})
        from .report import write_run_reports

        write_run_reports(context.store, context.config, context.validation_results)
    except Exception:
        pass


def _run_phases(context: _RunContext) -> DatasetState:
    store = context.store
    for phase in PHASE_ORDER:
        store.update_manifest(state="PROFILING")
        store.audit("phase_started", version_id=context.state.version_id, details={"phase": phase.value})
        while True:
            detection = detect_phase(context.state, context.config, phase)
            context.state = _merge_detection(context, detection)
            store.update_manifest(state="BUILDING_CANDIDATES")
            candidates = _actionable_candidates(context, detection)
            if not candidates:
                break
            allowed, reason = context.budget.can_attempt()
            if not allowed:
                context.state = _mark_unresolved(
                    context,
                    [issue.id for issue in detection.issues],
                    [ReasonCode.BUDGET_EXHAUSTED],
                    detail={"phase": phase.value, "budget_reason": reason},
                )
                context.budget_stop = True
                context.stop_reason = reason
                store.audit(
                    "budget_stop",
                    version_id=context.state.version_id,
                    details={"phase": phase.value, "reason": reason},
                )
                return context.state
            batch, sample_limit = _fit_batch(context, candidates)
            if batch is None:
                context.state = _mark_unresolved(
                    context,
                    [issue_id for candidate in candidates for issue_id in candidate.issue_ids],
                    [ReasonCode.REQUEST_TOO_LARGE],
                    detail={"phase": phase.value},
                )
                continue
            committed = _process_batch(context, phase, batch, sample_limit)
            if context.budget_stop:
                return context.state
            if not committed:
                store.update_manifest(state="REJECTED")
                if not _has_actionable(context, detection):
                    break
    return context.state


def _merge_detection(context: _RunContext, detection: PhaseDetection) -> DatasetState:
    state = context.state
    issues = {issue.id: issue for issue in state.issues}
    for evidence in detection.evidence:
        context.evidence[evidence.id] = evidence
    for issue in detection.issues:
        issues.setdefault(issue.id, issue)
    selections = dict(state.selections)
    selections.update(detection.selections)
    return state.model_copy(update={"issues": list(issues.values()), "selections": selections})


def _actionable_candidates(context: _RunContext, detection: PhaseDetection) -> list[RepairCandidate]:
    state = context.state
    status = {issue.id: issue.status for issue in state.issues}
    candidates = [
        candidate
        for candidate in build_candidates(state, context.config, detection)
        if all(status.get(issue_id) == IssueStatus.OPEN for issue_id in candidate.issue_ids)
    ]
    with_candidate = {issue_id for candidate in candidates for issue_id in candidate.issue_ids}
    missing = [
        issue
        for issue in detection.issues
        if issue.status == IssueStatus.OPEN and issue.id not in with_candidate
    ]
    if missing:
        context.state = _mark_unresolved(
            context,
            [issue.id for issue in missing],
            [ReasonCode.CANDIDATE_UNAVAILABLE],
            detail={"phase": detection.phase.value},
        )
        candidates = [
            candidate
            for candidate in candidates
            if all(
                issue.status == IssueStatus.OPEN
                for issue in context.state.issues
                if issue.id in candidate.issue_ids
            )
        ]
    return candidates


def _has_actionable(context: _RunContext, detection: PhaseDetection) -> bool:
    state = context.state
    status = {issue.id: issue.status for issue in state.issues}
    candidates = build_candidates(state, context.config, detection)
    return any(
        all(status.get(issue_id) == IssueStatus.OPEN for issue_id in candidate.issue_ids)
        for candidate in candidates
    )


def _fit_batch(
    context: _RunContext, candidates: list[RepairCandidate]
) -> tuple[list[RepairCandidate] | None, int]:
    """Reduce batch size, then samples, until the request fits the byte budget."""
    state = context.state
    config = context.config
    if not candidates:
        return None, 0
    role = OPERATION_ROLE[candidates[0].operation]
    for count in range(min(len(candidates), MAX_CANDIDATES_PER_BATCH), 0, -1):
        subset = candidates[:count]
        for sample_limit in (MAX_SAMPLES_PER_CANDIDATE, 4, 2, 1, 0):
            contexts = [
                build_candidate_context(
                    state, config, candidate, role, context.evidence, sample_limit=sample_limit
                )
                for candidate in subset
            ]
            questions = build_questions(role, contexts)
            payload = {
                "model": MODEL_NAME,
                "state": {"candidates": contexts},
                "questions": serialize_questions(questions),
            }
            size = len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))
            if size <= MAX_REQUEST_BYTES:
                if count < len(candidates) or sample_limit < MAX_SAMPLES_PER_CANDIDATE:
                    context.store.audit(
                        "request_truncated",
                        version_id=state.version_id,
                        details={
                            "request_bytes": size,
                            "candidates": count,
                            "sample_limit": sample_limit,
                        },
                    )
                return subset, sample_limit
    return None, 0


def _process_batch(
    context: _RunContext,
    phase: Phase,
    candidates: list[RepairCandidate],
    sample_limit: int,
) -> bool:
    store = context.store
    state = context.state
    role = OPERATION_ROLE[candidates[0].operation]
    agent = agent_for_role(role)
    store.update_manifest(state="DECIDING")
    result = agent.evaluate(
        state,
        context.config,
        candidates,
        context.evidence,
        context.client,
        sample_limit=sample_limit,
    )
    request_ref = _log_request(context, "decision", phase, result)
    # Point every decision at the request payload that actually holds its bounded state
    # and complete questions; the reference is resolved against the run directory.
    result.records = [
        record.model_copy(update={"context_ref": request_ref, "questions_ref": request_ref})
        for record in result.records
    ]
    for record in result.records:
        context.records.append(record)
    if result.outcome.model_status == ModelStatus.NOT_CALLED:
        context.budget_stop = True
        context.stop_reason = result.outcome.error or "BUDGET_EXHAUSTED"
        context.state = _mark_unresolved(
            context,
            [issue_id for candidate in candidates for issue_id in candidate.issue_ids],
            [ReasonCode.BUDGET_EXHAUSTED],
            detail={"request_ref": request_ref},
        )
        _write_decisions(context, result.records, committed=False)
        _write_traces_for_non_planned(
            context, result.records, None, [ReasonCode.BUDGET_EXHAUSTED]
        )
        return False

    # Rejected candidates are final for this snapshot: mark their issues unresolved so a
    # later version change cannot re-sample the same operation.
    for candidate, record in zip(candidates, result.records, strict=True):
        if not record.gate_accepted:
            context.state = _mark_unresolved(
                context,
                candidate.issue_ids,
                record.reason_codes or [ReasonCode.MODEL_ABSTAIN],
                detail={"stage": "gate", "request_ref": request_ref},
            )
    state = context.state

    accepted = [
        AcceptedCandidate(candidate=candidate, decision_id=record.id)
        for candidate, record in zip(candidates, result.records, strict=True)
        if record.gate_accepted
    ]
    store.update_manifest(state="PLANNING")
    plan = build_plan(state, context.config, context.evidence, accepted, context.next_plan())
    if not plan.ordered_operations:
        reasons = _gate_or_plan_reasons(context, result.records, plan, candidates)
        context.state = _mark_unresolved(
            context,
            [issue_id for candidate in candidates for issue_id in candidate.issue_ids],
            reasons,
            detail={"plan_id": plan.id, "stage": "planner"},
        )
        _write_decisions(context, result.records, committed=False)
        _write_traces_for_non_planned(context, result.records, plan, reasons)
        return False

    store.update_manifest(state="EXECUTING")
    try:
        execution = execute(state, plan, context.config, context.next_version())
    except ExecutionError as error:
        context.state = _mark_unresolved(
            context,
            [issue_id for candidate in candidates for issue_id in candidate.issue_ids],
            [ReasonCode.HARD_VALIDATION_FAILED],
            detail={"plan_id": plan.id, "error": str(error)},
        )
        _write_decisions(context, result.records, committed=False)
        _write_traces_for_non_planned(context, result.records, plan, [ReasonCode.HARD_VALIDATION_FAILED])
        return False

    store.update_manifest(state="VALIDATING")
    hard = validate_hard(state, execution, plan, context.config)
    if plan.discarded:
        hard.reason_codes = [*hard.reason_codes, ReasonCode.CANDIDATE_CONFLICT]
    context.validation_results.append(hard)
    store.write_validation(context.validation_results)
    if not hard.passed:
        context.state = _mark_unresolved(
            context,
            [issue_id for candidate in candidates for issue_id in candidate.issue_ids],
            [ReasonCode.HARD_VALIDATION_FAILED],
            detail={"plan_id": plan.id},
        )
        _write_decisions(context, result.records, committed=False)
        _write_traces_for_non_planned(
            context, result.records, plan, [ReasonCode.HARD_VALIDATION_FAILED], hard_status="failed"
        )
        store.audit(
            "plan_rejected",
            version_id=state.version_id,
            plan_id=plan.id,
            candidate_ids=plan.candidate_ids,
            details={"stage": "hard_validation"},
        )
        return False

    postcheck = _run_postcheck(context, phase, state, plan, execution.changes, candidates)
    if postcheck is None:
        context.state = _mark_unresolved(
            context,
            [issue_id for candidate in candidates for issue_id in candidate.issue_ids],
            [ReasonCode.REQUEST_TOO_LARGE],
            detail={"plan_id": plan.id},
        )
        _write_decisions(context, result.records, committed=False)
        _write_traces_for_non_planned(
            context, result.records, plan, [ReasonCode.REQUEST_TOO_LARGE]
        )
        return False
    semantic_checks = validate_semantic(plan, postcheck.outcome)
    hard.semantic_checks = semantic_checks
    hard.passed = hard.passed and all(check.passed for check in semantic_checks)
    store.write_validation(context.validation_results)
    if not hard.passed:
        failing = [check.candidate_id for check in semantic_checks if not check.passed]
        reasons = [ReasonCode.SEMANTIC_VALIDATION_FAILED]
        context.state = _mark_unresolved(
            context,
            [issue_id for candidate in candidates for issue_id in candidate.issue_ids],
            reasons,
            detail={"plan_id": plan.id, "failing_candidates": failing},
        )
        _write_decisions(context, result.records, committed=False)
        _write_traces_for_non_planned(
            context,
            result.records,
            plan,
            reasons,
            postcheck_status="failed",
            postcheck_outcome=postcheck,
            semantic_checks=semantic_checks,
            blocking=failing,
        )
        store.audit(
            "batch_rejected",
            version_id=state.version_id,
            plan_id=plan.id,
            candidate_ids=plan.candidate_ids,
            details={"stage": "semantic_validation", "failing_candidates": failing},
        )
        return False

    return _commit(context, phase, plan, execution, hard, result.records, candidates)


def _gate_or_plan_reasons(
    context: _RunContext,
    records: list[DecisionRecord],
    plan: RepairPlan,
    candidates: list[RepairCandidate],
) -> list[ReasonCode]:
    reasons: list[ReasonCode] = []
    for record in records:
        if not record.gate_accepted:
            reasons.extend(record.reason_codes)
    for discard in plan.discarded:
        try:
            reasons.append(ReasonCode(discard["reason_code"]))
        except ValueError:
            reasons.append(ReasonCode.CANDIDATE_UNAVAILABLE)
    if not reasons:
        reasons.append(ReasonCode.NO_CHANGE)
    return reasons


def _commit(
    context: _RunContext,
    phase: Phase,
    plan: RepairPlan,
    execution,
    hard: ValidationResult,
    records: list[DecisionRecord],
    candidates: list[RepairCandidate],
) -> bool:
    store = context.store
    state_before = context.state
    new_state = execution.state
    candidate_map = {candidate.id: candidate for candidate in candidates}
    committed_candidates = set(plan.candidate_ids)
    issue_map = {issue.id: issue for issue in new_state.issues}
    repaired_ids: set[str] = set()
    for candidate_id in committed_candidates:
        candidate = candidate_map.get(candidate_id)
        if candidate is None:
            continue
        issues = [issue_map[issue_id] for issue_id in candidate.issue_ids if issue_id in issue_map]
        selections = {issue.selection_id or "": new_state.selection(issue.selection_id or "") for issue in issues}
        selections = {key: value for key, value in selections.items() if value is not None}
        repaired = confirmed_repaired(new_state, context.config, phase, issues, selections)
        repaired_ids.update(repaired)
        for issue in issues:
            if issue.id in repaired:
                issue_map[issue.id] = issue.model_copy(
                    update={"status": IssueStatus.REPAIRED}
                )
            else:
                issue_map[issue.id] = issue.model_copy(
                    update={
                        "status": IssueStatus.UNRESOLVED,
                        "reason_codes": [*issue.reason_codes, ReasonCode.HARD_VALIDATION_FAILED],
                    }
                )
    for discard in plan.discarded:
        reason = _safe_reason(discard.get("reason_code"))
        for candidate_id in discard.get("candidate_ids", []):
            candidate = candidate_map.get(candidate_id)
            if candidate is None:
                continue
            for issue_id in candidate.issue_ids:
                issue = issue_map.get(issue_id)
                if issue is not None and issue.status != IssueStatus.REPAIRED:
                    issue_map[issue_id] = issue.model_copy(
                        update={
                            "status": IssueStatus.UNRESOLVED,
                            "reason_codes": [*issue.reason_codes, reason],
                        }
                    )
    new_state.issues = list(issue_map.values())
    store.save_snapshot(new_state)
    store.record_version(
        new_state,
        plan_id=plan.id,
        candidate_ids=list(committed_candidates),
        summary={
            "phase": phase.value,
            "changes": len(execution.changes),
            "removed_rows": len(execution.removed_rows),
        },
    )
    context.state = new_state
    store.update_manifest(state="COMMITTED")
    store.append_changes(
        plan.id,
        state_before.version_id,
        list(committed_candidates),
        execution.changes,
        committed=True,
        committed_version=new_state.version_id,
    )
    for record in records:
        context.store.append_decision(
            record, committed=record.candidate_id in committed_candidates
        )
    semantic_map = {check.candidate_id: check for check in hard.semantic_checks}
    for record in records:
        if record.candidate_id not in committed_candidates:
            rejected_trace = DecisionTrace(
                decision_id=record.id,
                candidate_id=record.candidate_id,
                plan_id=plan.id,
                planner_status="not_planned",
                commit_status="not_committed",
                reason_codes=record.reason_codes,
            )
            context.traces.append(rejected_trace)
            store.append_trace(rejected_trace)
            continue
        semantic = semantic_map.get(record.candidate_id)
        trace = DecisionTrace(
            decision_id=record.id,
            candidate_id=record.candidate_id,
            plan_id=plan.id,
            planner_status="planned",
            hard_validation_status="passed",
            postcheck_status="passed" if semantic and semantic.passed else "not_run",
            postcheck_probability=semantic.probability if semantic else None,
            commit_status="committed",
            committed_version=new_state.version_id,
        )
        context.traces.append(trace)
        store.append_trace(trace)
    store.audit(
        "plan_committed",
        version_id=new_state.version_id,
        plan_id=plan.id,
        candidate_ids=list(committed_candidates),
        details={
            "changes": len(execution.changes),
            "removed_rows": len(execution.removed_rows),
            "repaired_issues": sorted(repaired_ids),
        },
    )
    store.export_current()
    return True


def _run_postcheck(
    context: _RunContext,
    phase: Phase,
    state: DatasetState,
    plan: RepairPlan,
    changes,
    candidates: list[RepairCandidate],
) -> AgentBatchResult | None:
    candidate_map = {candidate.id: candidate for candidate in candidates}
    contexts: list[dict[str, Any]] = []
    questions: dict[str, Any] = {}
    for sample_limit in (MAX_SAMPLES_PER_CANDIDATE, 4, 2, 1, 0):
        contexts = build_postcheck_contexts(
            state, context.config, plan, changes, candidate_map, sample_limit=sample_limit
        )
        questions = build_postcheck_questions(contexts)
        payload = {
            "model": MODEL_NAME,
            "state": {"candidates": contexts},
            "questions": serialize_questions(questions),
        }
        size = len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))
        if size <= MAX_REQUEST_BYTES:
            if sample_limit < MAX_SAMPLES_PER_CANDIDATE:
                context.store.audit(
                    "request_truncated",
                    version_id=state.version_id,
                    details={"kind": "postcheck", "request_bytes": size, "sample_limit": sample_limit},
                )
            break
    else:
        return None
    outcome = context.client.evaluate({"candidates": contexts}, questions, kind="postcheck")
    _log_request(context, "postcheck", phase, None, contexts=contexts, questions=questions, outcome=outcome)
    return AgentBatchResult(role="semantic_postcheck", contexts=contexts, questions=questions, outcome=outcome)


def _log_request(
    context: _RunContext,
    kind: str,
    phase: Phase,
    result: AgentBatchResult | None,
    *,
    contexts: list[dict[str, Any]] | None = None,
    questions: dict[str, Any] | None = None,
    outcome: ClientOutcome | None = None,
) -> str:
    if result is not None:
        contexts = result.contexts
        questions = result.questions
        outcome = result.outcome
    raw_answers = None
    if outcome is not None and outcome.answers:
        raw_answers = {}
        for name, answer in outcome.answers.items():
            if hasattr(answer, "model_dump"):
                raw_answers[name] = answer.model_dump(mode="json")
            else:
                raw_answers[name] = answer
    serialized_questions = serialize_questions(questions or {})
    payload = {
        "kind": kind,
        "phase": phase.value,
        "version_id": context.state.version_id,
        "model": outcome.model if outcome else MODEL_NAME,
        "request_id": outcome.request_id if outcome else None,
        "model_status": outcome.model_status.value if outcome else "not_called",
        "question_version": QUESTION_VERSION,
        "state": {"candidates": contexts or []},
        "questions": serialized_questions,
        "context_hash": fingerprint(contexts or []),
        "questions_hash": fingerprint(serialized_questions),
        "candidate_hashes": {
            context_payload.get("candidate_id", ""): fingerprint(context_payload)
            for context_payload in (contexts or [])
        },
        "raw_answers": raw_answers,
        "usage": outcome.usage.model_dump() if outcome and outcome.usage else None,
        "attempts": [attempt.model_dump() for attempt in outcome.attempts] if outcome else [],
        "elapsed_seconds": outcome.elapsed_seconds if outcome else 0.0,
        "error": outcome.error if outcome else None,
        "invalid_details": outcome.invalid_details if outcome else None,
        "budget": context.budget.snapshot(),
    }
    sequence = context.next_request()
    return context.store.write_request(sequence, payload)


def _write_decisions(context: _RunContext, records: list[DecisionRecord], *, committed: bool) -> None:
    for record in records:
        context.store.append_decision(record, committed=committed)


def _write_traces_for_non_planned(
    context: _RunContext,
    records: list[DecisionRecord],
    plan: RepairPlan | None,
    reasons: list[ReasonCode],
    *,
    hard_status: str = "not_run",
    postcheck_status: str = "not_run",
    postcheck_outcome: ClientOutcome | None = None,
    semantic_checks=None,
    blocking: list[str] | None = None,
) -> None:
    semantic_map = {check.candidate_id: check for check in (semantic_checks or [])}
    planned = set(plan.candidate_ids) if plan is not None else set()
    for record in records:
        if record.candidate_id in planned:
            planner_status = "planned"
        elif record.gate_accepted:
            planner_status = "discarded"
        else:
            planner_status = "not_planned"
        semantic = semantic_map.get(record.candidate_id)
        trace_reasons = list(reasons)
        if blocking and record.candidate_id not in blocking and record.candidate_id in planned:
            trace_reasons = [ReasonCode.BATCH_REJECTED]
        trace = DecisionTrace(
            decision_id=record.id,
            candidate_id=record.candidate_id,
            plan_id=plan.id if plan else None,
            planner_status=planner_status,
            hard_validation_status=hard_status,
            postcheck_status=postcheck_status,
            postcheck_probability=semantic.probability if semantic else None,
            commit_status="not_committed",
            blocking_candidate_ids=blocking or [],
            reason_codes=trace_reasons,
        )
        context.traces.append(trace)
        context.store.append_trace(trace)


def _mark_unresolved(
    context: _RunContext,
    issue_ids: list[str],
    reasons: list[ReasonCode],
    *,
    detail: dict[str, Any] | None = None,
) -> DatasetState:
    state = context.state
    if not issue_ids:
        return state
    target = set(issue_ids)
    issues = []
    for issue in state.issues:
        if issue.id in target and issue.status == IssueStatus.OPEN:
            issues.append(
                issue.model_copy(
                    update={"status": IssueStatus.UNRESOLVED, "reason_codes": [*issue.reason_codes, *reasons]}
                )
            )
        else:
            issues.append(issue)
    return state.model_copy(update={"issues": issues})


def _safe_reason(value: Any) -> ReasonCode:
    try:
        return ReasonCode(value)
    except (ValueError, TypeError):
        return ReasonCode.CANDIDATE_UNAVAILABLE


def _finalize(context: _RunContext) -> RunOutcome:
    state = context.state
    store = context.store
    store.save_snapshot(state)
    store.export_current()
    unresolved = [issue for issue in state.issues if issue.status == IssueStatus.UNRESOLVED]
    open_issues = [issue for issue in state.issues if issue.status == IssueStatus.OPEN]
    if context.budget_stop:
        outcome = RunOutcome.PARTIAL
        context.stop_reason = context.stop_reason or "budget_exhausted"
    elif unresolved or open_issues:
        outcome = RunOutcome.PARTIAL
        context.stop_reason = context.stop_reason or "unresolved_issues"
    else:
        outcome = RunOutcome.COMPLETED
        context.stop_reason = context.stop_reason or "no_actionable_issues"
    mark_outcome(store, outcome, context.stop_reason, "FINISHED")
    store.audit(
        "run_finished",
        version_id=state.version_id,
        details={
            "outcome": outcome.value,
            "stop_reason": context.stop_reason,
            "unresolved_issues": len(unresolved),
            "open_issues": len(open_issues),
            "budget": context.budget.snapshot(),
        },
    )
    from .report import write_run_reports

    write_run_reports(store, context.config, context.validation_results)
    return outcome
