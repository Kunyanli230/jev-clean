"""Read-only inspection of the input snapshot using the cleaning detectors."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .candidates import build_candidates
from .config import config_hash, load_config
from .evaluation import quality_counts
from .models import PHASE_ORDER
from .profiling import detect_phase
from .storage import load_csv


def profile_csv(input_path: str | Path, config_path: str | Path) -> dict[str, Any]:
    """Describe current data and authorized candidate previews without applying repairs.

    All phases see the same original cells. Earlier unresolved issues are retained
    so dependency blocking is visible; previews are independent possibilities,
    not a combined repair plan or predictions about later cleaned snapshots.
    """
    config = load_config(config_path)
    state = load_csv(input_path, config)
    column_index = state.column_index()
    quality = quality_counts(state, config)
    phases = []
    issues = {}
    for phase in PHASE_ORDER:
        detection = detect_phase(state, config, phase)
        issues.update({issue.id: issue for issue in detection.issues})
        state = state.model_copy(
            update={
                "issues": list(issues.values()),
                "selections": {**state.selections, **detection.selections},
            }
        )
        candidates = build_candidates(state, config, detection)
        phases.append(
            {
                "phase": phase.value,
                "issues": [issue.model_dump(mode="json") for issue in detection.issues],
                "evidence": [entry.model_dump(mode="json") for entry in detection.evidence],
                "candidates": [
                    {
                        "id": candidate.id,
                        "operation": candidate.operation.value,
                        "column": state.selections[candidate.selection_id].column,
                        "target_count": len(state.selections[candidate.selection_id].row_ids),
                        "issue_ids": candidate.issue_ids,
                        "preview": [cell.model_dump(mode="json") for cell in candidate.preview],
                    }
                    for candidate in candidates
                ],
            }
        )
    return {
        "kind": "input_profile",
        "artifact_version": 1,
        "model_identity": "none",
        "model_calls": 0,
        "input_hash": state.input_hash,
        "config_hash": config_hash(config),
        "snapshot": state.version_id,
        "rows": len(state.rows),
        "columns": state.ordered_columns,
        "quality": quality,
        "declared_null_token_cells": sum(
            row[column_index[name]] in column.null_tokens
            for row in state.rows
            for name, column in config.columns.items()
        ),
        "issue_counts": dict(Counter(issue.category.value for issue in issues.values())),
        "candidate_count": sum(len(phase["candidates"]) for phase in phases),
        "candidate_semantics": (
            "Independent previews on the unchanged input; no gate approval or combined plan. "
            "Later candidates may become available after earlier repairs."
        ),
        "phases": phases,
    }
