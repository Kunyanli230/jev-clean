"""Acceptance 9, 10: retry/budget behavior at the SDK boundary and no-key safety."""

from __future__ import annotations

import httpx2
import pytest
from fake_client import write_config, write_csv_file
from typesafe_sdk import Noul, RetryPolicy, TypeSafeClient

from idac.models import ModelStatus
from idac.typesafe_client import (
    ConfigurationError,
    RunBudget,
    TypeSafeDecisionClient,
)

SUCCESS_BODY = {
    "model": "jev-1.13.0",
    "usage": {"input_tokens": 100, "output_tokens": 5},
    "answers": {"q": {"type": "noul", "noul": 0.9}},
}

CONFIG_TEXT = """
table_description: "client test"
record_grain: "one row per record"
columns:
  id:
    type: string
    description: "identifier"
    protected: true
    null_tokens: [""]
    allowed_operations: []
duplicates:
  enabled: false
  compare_columns: []
"""


def make_client(handler, budget: RunBudget | None = None) -> TypeSafeDecisionClient:
    sdk = TypeSafeClient(
        api_key="test-key",
        base_url="https://api.test",
        model="jev-1.13.0",
        timeout=5.0,
        transport=httpx2.MockTransport(handler),
        retry=RetryPolicy(max_retries=0),  # IDAC owns the only retry layer
    )
    return TypeSafeDecisionClient(budget=budget or RunBudget(), client=sdk)


def questions() -> dict:
    return {"q": Noul(instructions="Is this a test?")}


def test_401_raises_configuration_error_and_stops() -> None:
    calls = []

    def handler(request):
        calls.append(request)
        return httpx2.Response(401, json={"error": "bad key"}, headers={"x-typesafe-request-id": "r1"})

    client = make_client(handler)
    with pytest.raises(ConfigurationError):
        client.evaluate({"a": 1}, questions())
    assert len(calls) == 1


def test_422_is_not_retried() -> None:
    calls = []

    def handler(request):
        calls.append(request)
        return httpx2.Response(422, json={"detail": "invalid"})

    outcome = make_client(handler).evaluate({"a": 1}, questions())
    assert outcome.model_status == ModelStatus.FAILED
    assert len(calls) == 1


def test_429_retries_once_respecting_retry_after() -> None:
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx2.Response(429, json={"error": "slow down"}, headers={"retry-after": "0"})
        return httpx2.Response(200, json=SUCCESS_BODY)

    outcome = make_client(handler).evaluate({"a": 1}, questions())
    assert outcome.model_status == ModelStatus.OK
    assert len(calls) == 2
    assert [attempt.status for attempt in outcome.attempts] == ["error", "ok"]


def test_529_retries_once_and_no_hidden_sdk_retry(monkeypatch) -> None:
    monkeypatch.setattr("idac.typesafe_client.RETRY_BACKOFF_SECONDS", 0.0)
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx2.Response(529, json={"error": "overloaded"})
        return httpx2.Response(200, json=SUCCESS_BODY)

    outcome = make_client(handler).evaluate({"a": 1}, questions())
    assert outcome.model_status == ModelStatus.OK
    assert len(calls) == 2

    always = []

    def always_529(request):
        always.append(request)
        return httpx2.Response(529, json={"error": "overloaded"})

    outcome = make_client(always_529).evaluate({"a": 1}, questions())
    assert outcome.model_status == ModelStatus.FAILED
    assert len(always) == 2  # one retry, never more


def test_timeout_retries_once_then_fails(monkeypatch) -> None:
    monkeypatch.setattr("idac.typesafe_client.RETRY_BACKOFF_SECONDS", 0.0)
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx2.TimeoutException("too slow")

    outcome = make_client(handler).evaluate({"a": 1}, questions())
    assert outcome.model_status == ModelStatus.FAILED
    assert len(calls) == 2


def test_connection_error_retries_once_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr("idac.typesafe_client.RETRY_BACKOFF_SECONDS", 0.0)
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx2.ConnectError("no route")
        return httpx2.Response(200, json=SUCCESS_BODY)

    outcome = make_client(handler).evaluate({"a": 1}, questions())
    assert outcome.model_status == ModelStatus.OK
    assert len(calls) == 2


def test_attempt_budget_stops_before_request() -> None:
    calls = []

    def handler(request):
        calls.append(request)
        return httpx2.Response(200, json=SUCCESS_BODY)

    budget = RunBudget(max_attempts=1)
    client = make_client(handler, budget)
    first = client.evaluate({"a": 1}, questions())
    assert first.model_status == ModelStatus.OK
    second = client.evaluate({"a": 1}, questions())
    assert second.model_status == ModelStatus.NOT_CALLED
    assert second.error == "MAX_SDK_REQUEST_ATTEMPTS"
    assert len(calls) == 1


def test_input_token_budget_stops_next_request() -> None:
    def handler(request):
        return httpx2.Response(200, json=SUCCESS_BODY)

    budget = RunBudget(max_input_tokens=50)
    client = make_client(handler, budget)
    first = client.evaluate({"a": 1}, questions())
    assert first.model_status == ModelStatus.OK
    second = client.evaluate({"a": 1}, questions())
    assert second.model_status == ModelStatus.NOT_CALLED
    assert second.error == "MAX_INPUT_TOKENS"


def test_unknown_usage_is_not_counted_as_zero() -> None:
    body = {
        "model": "jev-1.13.0",
        "usage": {},
        "answers": {"q": {"type": "noul", "noul": 0.9}},
    }

    def handler(request):
        return httpx2.Response(200, json=body)

    budget = RunBudget()
    client = make_client(handler, budget)
    outcome = client.evaluate({"a": 1}, questions())
    assert outcome.usage is not None and outcome.usage.known is False
    assert budget.input_tokens == 0
    assert budget.unknown_usage_requests == 1


def test_invalid_response_body_is_invalid_response() -> None:
    def handler(request):
        return httpx2.Response(200, json={"model": "jev-1.13.0", "answers": {"q": {"type": "noul", "noul": "high"}}})

    outcome = make_client(handler).evaluate({"a": 1}, questions())
    assert outcome.model_status == ModelStatus.INVALID_RESPONSE
    assert outcome.invalid_details is not None


def test_missing_api_key_does_not_create_output_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    from typesafe_sdk import TypeSafeError

    from idac.orchestrator import run_clean

    config_path = write_config(tmp_path, CONFIG_TEXT)
    input_path = write_csv_file(tmp_path, ["id"], [["001"]])
    output = tmp_path / "run"
    with pytest.raises(TypeSafeError):
        run_clean(input_path, config_path, output)
    assert not output.exists()


def test_clean_cli_without_key_does_not_create_run(tmp_path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from idac.cli import app

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    config_path = write_config(tmp_path, CONFIG_TEXT)
    input_path = write_csv_file(tmp_path, ["id"], [["001"]])
    output = tmp_path / "run"
    result = CliRunner().invoke(
        app, ["clean", "--input", str(input_path), "--config", str(config_path), "--output", str(output)]
    )
    assert result.exit_code == 1
    assert "TYPESAFE_API_KEY" in result.output
    assert not output.exists()


def test_clean_refuses_existing_output(tmp_path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from idac.cli import app

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    config_path = write_config(tmp_path, CONFIG_TEXT)
    input_path = write_csv_file(tmp_path, ["id"], [["001"]])
    output = tmp_path / "run"
    output.mkdir()
    result = CliRunner().invoke(
        app, ["clean", "--input", str(input_path), "--config", str(config_path), "--output", str(output)]
    )
    assert result.exit_code == 1
    assert "already exists" in result.output
