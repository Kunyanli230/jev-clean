"""Run storage: CSV loading, JSON snapshots, append-only audit logs and exports.

The original CSV is never overwritten. A run directory contains the manifest,
snapshots, append-only decision/change/audit logs, request payloads and the
exports generated from the manifest's current version pointer.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import TableConfig, config_hash, load_config
from .models import (
    AuditEvent,
    CellChange,
    DatasetState,
    DecisionRecord,
    DecisionTrace,
    Issue,
    RemovedRow,
    RunOutcome,
    ValidationResult,
    sha256_bytes,
)
from .settings import MAX_COLUMNS, MAX_ROWS, MODEL_NAME, POLICY_VERSION, QUESTION_VERSION

RUN_FILES = {
    "manifest": "manifest.json",
    "config": "config.yaml",
    "cleaned": "cleaned.csv",
    "removed": "removed_rows.csv",
    "issues": "issues.json",
    "decisions": "decisions.jsonl",
    "traces": "decision_traces.jsonl",
    "changes": "changes.jsonl",
    "audit": "audit.jsonl",
    "validation": "validation.json",
    "report": "report.md",
    "cards": "decision_cards.md",
    "requests": "requests",
    "snapshots": "snapshots",
}


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write a file through a temporary sibling and an atomic replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix=target.name, suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, target)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_name)
        raise


def atomic_to_csv(frame: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix=target.name, suffix=".tmp"
    )
    os.close(descriptor)
    try:
        frame.to_csv(tmp_name, **kwargs)
        os.replace(tmp_name, target)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_name)
        raise


class StorageError(RuntimeError):
    pass


def load_csv(path: str | Path, config: TableConfig) -> DatasetState:
    """Load the input CSV as strings; auto NA inference is disabled."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise StorageError(f"input CSV not found: {csv_path}")
    raw_bytes = csv_path.read_bytes()
    try:
        frame = pd.read_csv(
            csv_path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8",
        )
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise StorageError(f"could not parse input CSV: {error}") from error
    if len(frame) > MAX_ROWS:
        raise StorageError(f"input has {len(frame)} rows, limit is {MAX_ROWS}")
    if len(frame.columns) > MAX_COLUMNS:
        raise StorageError(f"input has {len(frame.columns)} columns, limit is {MAX_COLUMNS}")
    missing = [column for column in frame.columns if column not in config.columns]
    if missing:
        raise StorageError(f"columns missing from configuration: {missing}")
    extra = [column for column in config.columns if column not in set(frame.columns)]
    if extra:
        raise StorageError(f"configured columns missing from input: {extra}")
    ordered_columns = [str(column) for column in frame.columns]
    rows = [[value if isinstance(value, str) else None for value in row] for row in frame.values.tolist()]
    row_ids = [f"row-{index:06d}" for index in range(len(rows))]
    return DatasetState(
        version_id="v000",
        parent_version_id=None,
        input_hash=sha256_bytes(raw_bytes),
        ordered_columns=ordered_columns,
        row_ids=row_ids,
        rows=rows,
        schema=config.logical_schema(),
        removed_rows=[],
        issues=[],
    )


def _frame(state: DatasetState) -> pd.DataFrame:
    return pd.DataFrame(state.rows, columns=state.ordered_columns, dtype=object)


def write_cleaned_csv(state: DatasetState, path: str | Path) -> None:
    atomic_to_csv(_frame(state), path, index=False, na_rep="", lineterminator="\n")


def write_removed_csv(state: DatasetState, path: str | Path) -> None:
    header = ["row_id", "reason", "candidate_id", *state.ordered_columns]
    lines = [",".join(header)]
    for removed in state.removed_rows:
        values = [removed.row_id, removed.reason, removed.candidate_id or ""]
        values.extend("" if cell is None else cell for cell in removed.values)
        lines.append(",".join(_csv_cell(value) for value in values))
    atomic_write_text(path, "\n".join(lines) + "\n")


def _csv_cell(value: str) -> str:
    if any(character in value for character in [",", '"', "\n", "\r"]):
        return '"' + value.replace('"', '""') + '"'
    return value


class RunStore:
    """File-backed store for one clean run."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.manifest_path = self.run_dir / RUN_FILES["manifest"]
        self.snapshots_dir = self.run_dir / RUN_FILES["snapshots"]
        self.requests_dir = self.run_dir / RUN_FILES["requests"]

    @classmethod
    def create(
        cls,
        run_dir: str | Path,
        config_path: str | Path,
        input_path: str | Path,
        config: TableConfig,
        model_identity: str,
    ) -> RunStore:
        directory = Path(run_dir)
        if directory.exists():
            raise StorageError(f"output directory already exists: {directory}")
        directory.mkdir(parents=True)
        store = cls(directory)
        store.snapshots_dir.mkdir()
        store.requests_dir.mkdir()
        shutil.copyfile(config_path, store.run_dir / RUN_FILES["config"])
        manifest = {
            "run_id": directory.name,
            "created_at": now_iso(),
            "input_path": str(input_path),
            "input_hash": None,
            "config_hash": config_hash(config),
            "model": MODEL_NAME,
            "model_identity": model_identity,
            "policy_version": POLICY_VERSION,
            "question_version": QUESTION_VERSION,
            "current_version": "v000",
            "versions": {},
            "outcome": None,
            "stop_reason": None,
            "state": "LOADED",
            "finished": False,
        }
        store._write_manifest(manifest)
        return store

    # ------------------------------------------------------------------ manifest

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        atomic_write_text(
            self.manifest_path,
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        )

    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def update_manifest(self, **updates: Any) -> dict[str, Any]:
        manifest = self.manifest()
        manifest.update(updates)
        self._write_manifest(manifest)
        return manifest

    def record_version(
        self,
        state: DatasetState,
        *,
        plan_id: str | None,
        candidate_ids: list[str],
        summary: dict[str, Any] | None = None,
    ) -> None:
        manifest = self.manifest()
        manifest["versions"][state.version_id] = {
            "parent": state.parent_version_id,
            "created_at": now_iso(),
            "plan_id": plan_id,
            "candidate_ids": candidate_ids,
            "summary": summary or {},
        }
        manifest["current_version"] = state.version_id
        manifest["input_hash"] = state.input_hash
        self._write_manifest(manifest)

    def ancestor_chain(self, version_id: str | None = None) -> list[str]:
        manifest = self.manifest()
        current = version_id or manifest["current_version"]
        chain: list[str] = []
        seen: set[str] = set()
        while current is not None and current not in seen:
            seen.add(current)
            chain.append(current)
            entry = manifest["versions"].get(current)
            current = entry.get("parent") if entry else None
        chain.reverse()
        return chain

    # ------------------------------------------------------------------ snapshots

    def save_snapshot(self, state: DatasetState) -> None:
        path = self.snapshots_dir / f"{state.version_id}.json"
        atomic_write_text(path, state.model_dump_json(indent=2) + "\n")

    def load_snapshot(self, version_id: str) -> DatasetState:
        path = self.snapshots_dir / f"{version_id}.json"
        if not path.is_file():
            raise StorageError(f"snapshot not found: {version_id}")
        return DatasetState.model_validate_json(path.read_text(encoding="utf-8"))

    def current_state(self) -> DatasetState:
        return self.load_snapshot(self.manifest()["current_version"])

    # ------------------------------------------------------------------ appends

    def _append(self, name: str, payload: dict[str, Any]) -> None:
        path = self.run_dir / RUN_FILES[name]
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def append_decision(self, record: DecisionRecord, *, committed: bool) -> None:
        payload = record.model_dump(mode="json")
        payload["committed"] = committed
        self._append("decisions", payload)

    def append_trace(self, trace: DecisionTrace) -> None:
        self._append("traces", trace.model_dump(mode="json"))

    def append_changes(
        self,
        plan_id: str,
        version_id: str,
        candidate_ids: list[str],
        changes: list[CellChange],
        *,
        committed: bool,
        committed_version: str | None = None,
    ) -> None:
        self._append(
            "changes",
            {
                "timestamp": now_iso(),
                "plan_id": plan_id,
                "version_id": version_id,
                "candidate_ids": candidate_ids,
                "committed": committed,
                "committed_version": committed_version,
                "changes": [change.model_dump(mode="json") for change in changes],
            },
        )

    def append_audit(self, event: AuditEvent) -> None:
        self._append("audit", event.model_dump(mode="json"))

    def audit(self, event_type: str, **details: Any) -> AuditEvent:
        explicit = details.pop("details", None)
        event = AuditEvent(
            timestamp=now_iso(),
            event_type=event_type,
            version_id=details.pop("version_id", None),
            plan_id=details.pop("plan_id", None),
            candidate_ids=details.pop("candidate_ids", []),
            details={**details, **(explicit or {})},
        )
        self.append_audit(event)
        return event

    # ------------------------------------------------------------------ requests

    def write_request(self, sequence: int, payload: dict[str, Any]) -> str:
        name = f"req-{sequence:04d}.json"
        path = self.requests_dir / name
        atomic_write_text(
            path, json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
        )
        return f"requests/{name}"

    def read_requests(self) -> list[dict[str, Any]]:
        entries = []
        for path in sorted(self.requests_dir.glob("req-*.json")):
            entries.append(json.loads(path.read_text(encoding="utf-8")))
        return entries

    # ------------------------------------------------------------------ readers

    def _read_jsonl(self, name: str) -> list[dict[str, Any]]:
        path = self.run_dir / RUN_FILES[name]
        if not path.is_file():
            return []
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
        return entries

    def read_decisions(self) -> list[dict[str, Any]]:
        return self._read_jsonl("decisions")

    def read_traces(self) -> list[dict[str, Any]]:
        return self._read_jsonl("traces")

    def read_changes(self) -> list[dict[str, Any]]:
        return self._read_jsonl("changes")

    def read_audit(self) -> list[dict[str, Any]]:
        return self._read_jsonl("audit")

    # ------------------------------------------------------------------ validation / evaluation

    def write_validation(self, results: list[ValidationResult]) -> None:
        path = self.run_dir / RUN_FILES["validation"]
        atomic_write_text(
            path,
            json.dumps(
                [result.model_dump(mode="json") for result in results],
                indent=2,
                ensure_ascii=False,
                default=str,
            )
            + "\n",
        )

    def read_validation_results(self) -> list[dict[str, Any]]:
        path = self.run_dir / RUN_FILES["validation"]
        if not path.is_file():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def evaluation_path(self, version_id: str) -> Path:
        return self.run_dir / f"evaluation_{version_id}.json"

    def write_evaluation(self, version_id: str, payload: dict[str, Any]) -> Path:
        path = self.evaluation_path(version_id)
        atomic_write_text(
            path, json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
        )
        return path

    def read_evaluation(self, version_id: str) -> dict[str, Any] | None:
        path = self.evaluation_path(version_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ exports

    def export_current(self) -> DatasetState:
        state = self.current_state()
        write_cleaned_csv(state, self.run_dir / RUN_FILES["cleaned"])
        write_removed_csv(state, self.run_dir / RUN_FILES["removed"])
        atomic_write_text(
            self.run_dir / RUN_FILES["issues"],
            json.dumps(
                [issue.model_dump(mode="json") for issue in state.issues],
                indent=2,
                ensure_ascii=False,
                default=str,
            )
            + "\n",
        )
        return state

    def rollback(self, to_version: str) -> DatasetState:
        """Restore a committed snapshot and its exports; never calls the model."""
        manifest = self.manifest()
        if to_version not in manifest["versions"]:
            raise StorageError(f"version not committed in this run: {to_version}")
        if manifest["versions"][to_version]["parent"] is None and to_version != "v000":
            raise StorageError(f"version has no snapshot: {to_version}")
        state = self.load_snapshot(to_version)
        manifest["current_version"] = to_version
        manifest["state"] = "COMMITTED"
        self._write_manifest(manifest)
        self.export_current()
        self.audit("rollback", version_id=to_version, details={"to_version": to_version})
        return state


def load_run_config(store: RunStore) -> TableConfig:
    return load_config(store.run_dir / RUN_FILES["config"])


def write_issues(path: str | Path, issues: list[Issue]) -> None:
    atomic_write_text(
        path,
        json.dumps(
            [issue.model_dump(mode="json") for issue in issues],
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
    )


def load_truth_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8")


def summarize_state(state: DatasetState) -> dict[str, Any]:
    return {
        "version_id": state.version_id,
        "row_count": len(state.rows),
        "removed_count": len(state.removed_rows),
        "issue_count": len(state.issues),
    }


def mark_outcome(store: RunStore, outcome: RunOutcome, stop_reason: str, state: str) -> None:
    store.update_manifest(outcome=outcome.value, stop_reason=stop_reason, state=state, finished=True)


def removed_rows_from_ids(
    state: DatasetState, row_ids: list[str], reason: str, candidate_id: str | None
) -> list[RemovedRow]:
    index = state.row_index()
    removed = []
    for row_id in row_ids:
        position = index[row_id]
        removed.append(
            RemovedRow(
                row_id=row_id,
                values=list(state.rows[position]),
                reason=reason,
                candidate_id=candidate_id,
            )
        )
    return removed
