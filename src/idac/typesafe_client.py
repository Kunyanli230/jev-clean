"""TypeSafe SDK wrapper: the single retry layer, budget accounting and error mapping.

The SDK's own retries are disabled (``RetryPolicy(max_retries=0)``) so every
actual HTTP attempt is counted and budget-checked here. 429, 529, network errors
and timeouts are retried at most once; 401 aborts the run; 422 and invalid
responses are non-retryable request failures.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from typesafe_sdk import (
    RetryPolicy,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeClient,
    TypeSafeError,
)

from .models import ClientOutcome, ModelStatus, RequestAttempt, UsageInfo
from .settings import (
    API_KEY_ENV,
    BASE_URL,
    MAX_INPUT_TOKENS,
    MAX_SDK_REQUEST_ATTEMPTS,
    MODEL_NAME,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_BACKOFF_SECONDS,
    RETRY_MAX_ATTEMPTS,
    RUN_TIME_LIMIT_SECONDS,
)

RETRYABLE_STATUSES = {429, 529}


class ConfigurationError(RuntimeError):
    """Authentication or configuration failure that must stop the run."""


class DecisionClient(Protocol):
    model_identity: str

    def evaluate(
        self, state: dict[str, Any], questions: dict[str, Any], *, kind: str = "decision"
    ) -> ClientOutcome: ...

    def close(self) -> None: ...


class RunBudget:
    """Tracks actual attempts, known input tokens and wall-clock time."""

    def __init__(
        self,
        max_attempts: int = MAX_SDK_REQUEST_ATTEMPTS,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        time_limit: float = RUN_TIME_LIMIT_SECONDS,
    ) -> None:
        self.max_attempts = max_attempts
        self.max_input_tokens = max_input_tokens
        self.time_limit = time_limit
        self.started = time.monotonic()
        self.attempts = 0
        self.input_tokens = 0
        self.unknown_usage_requests = 0
        self.stop_reason: str | None = None

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def remaining_seconds(self) -> float:
        return max(0.0, self.time_limit - self.elapsed())

    def can_attempt(self) -> tuple[bool, str]:
        if self.stop_reason is not None:
            return False, self.stop_reason
        if self.remaining_seconds() <= 0.0:
            self.stop_reason = "RUN_TIME_LIMIT"
            return False, self.stop_reason
        if self.attempts >= self.max_attempts:
            self.stop_reason = "MAX_SDK_REQUEST_ATTEMPTS"
            return False, self.stop_reason
        if self.input_tokens >= self.max_input_tokens:
            self.stop_reason = "MAX_INPUT_TOKENS"
            return False, self.stop_reason
        return True, ""

    def register_attempt(self) -> None:
        self.attempts += 1

    def register_usage(self, input_tokens: int | None, output_tokens: int | None) -> None:
        if input_tokens is None:
            self.unknown_usage_requests += 1
        else:
            self.input_tokens += int(input_tokens)

    def snapshot(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "input_tokens": self.input_tokens,
            "max_input_tokens": self.max_input_tokens,
            "unknown_usage_requests": self.unknown_usage_requests,
            "elapsed_seconds": round(self.elapsed(), 3),
            "time_limit_seconds": self.time_limit,
            "stop_reason": self.stop_reason,
        }


def _retry_after_seconds(error: TypeSafeAPIError) -> float | None:
    value = getattr(error, "retry_after_ms", None)
    if isinstance(value, (int, float)) and value >= 0:
        return float(value) / 1000.0
    header = error.headers.get("retry-after") if error.headers is not None else None
    if header is not None:
        try:
            parsed = float(header.strip())
        except ValueError:
            return None
        if parsed >= 0:
            return parsed
    return None


class TypeSafeDecisionClient:
    """Synchronous client with the project's unique retry and budget layer."""

    def __init__(
        self,
        budget: RunBudget | None = None,
        *,
        api_key: str | None = None,
        model: str = MODEL_NAME,
        base_url: str = BASE_URL,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        client: TypeSafeClient | None = None,
    ) -> None:
        self.budget = budget or RunBudget()
        self.model_identity = "real"
        self._model = model
        self._client = client or TypeSafeClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            retry=RetryPolicy(max_retries=0),
        )

    def close(self) -> None:
        self._client.close()

    def evaluate(
        self, state: dict[str, Any], questions: dict[str, Any], *, kind: str = "decision"
    ) -> ClientOutcome:
        attempts: list[RequestAttempt] = []
        started = time.monotonic()
        for attempt_number in range(1, RETRY_MAX_ATTEMPTS + 1):
            allowed, reason = self.budget.can_attempt()
            if not allowed:
                return ClientOutcome(
                    model_status=ModelStatus.NOT_CALLED,
                    model=self._model,
                    error=reason,
                    attempts=attempts,
                    elapsed_seconds=time.monotonic() - started,
                )
            timeout = min(REQUEST_TIMEOUT_SECONDS, self.budget.remaining_seconds())
            if timeout <= 0:
                self.budget.stop_reason = "RUN_TIME_LIMIT"
                return ClientOutcome(
                    model_status=ModelStatus.NOT_CALLED,
                    model=self._model,
                    error="RUN_TIME_LIMIT",
                    attempts=attempts,
                    elapsed_seconds=time.monotonic() - started,
                )
            self.budget.register_attempt()
            try:
                response = self._client.system_one(
                    state=state, questions=questions, timeout=timeout
                )
            except TypeSafeAuthenticationError as error:
                raise ConfigurationError(
                    f"TypeSafe authentication failed ({error.status}); check {API_KEY_ENV}"
                ) from error
            except TypeSafeAPIResponseValidationError as error:
                attempts.append(
                    RequestAttempt(attempt=attempt_number, status="invalid_response", error=str(error))
                )
                return ClientOutcome(
                    model_status=ModelStatus.INVALID_RESPONSE,
                    model=self._model,
                    request_id=error.request_id,
                    error=str(error),
                    invalid_details={"field_path": error.field_path},
                    attempts=attempts,
                    elapsed_seconds=time.monotonic() - started,
                )
            except TypeSafeAPIError as error:
                retry_after = _retry_after_seconds(error)
                attempts.append(
                    RequestAttempt(
                        attempt=attempt_number,
                        status="error",
                        error=f"{error.status} {error}",
                        retry_after_seconds=retry_after,
                    )
                )
                if error.status in RETRYABLE_STATUSES and attempt_number < RETRY_MAX_ATTEMPTS:
                    wait = retry_after if retry_after is not None else RETRY_BACKOFF_SECONDS
                    if wait > self.budget.remaining_seconds():
                        self.budget.stop_reason = "RUN_TIME_LIMIT"
                        return ClientOutcome(
                            model_status=ModelStatus.FAILED,
                            model=self._model,
                            request_id=error.request_id,
                            error=f"retry wait {wait:.1f}s exceeds remaining run time",
                            attempts=attempts,
                            elapsed_seconds=time.monotonic() - started,
                        )
                    time.sleep(wait)
                    continue
                return ClientOutcome(
                    model_status=ModelStatus.FAILED,
                    model=self._model,
                    request_id=error.request_id,
                    error=str(error),
                    attempts=attempts,
                    elapsed_seconds=time.monotonic() - started,
                )
            except (TypeSafeAPITimeoutError, TypeSafeAPIConnectionError) as error:
                attempts.append(
                    RequestAttempt(attempt=attempt_number, status="connection_error", error=str(error))
                )
                if attempt_number < RETRY_MAX_ATTEMPTS:
                    if self.budget.remaining_seconds() < RETRY_BACKOFF_SECONDS:
                        self.budget.stop_reason = "RUN_TIME_LIMIT"
                        return ClientOutcome(
                            model_status=ModelStatus.FAILED,
                            model=self._model,
                            error="RUN_TIME_LIMIT",
                            attempts=attempts,
                            elapsed_seconds=time.monotonic() - started,
                        )
                    time.sleep(RETRY_BACKOFF_SECONDS)
                    continue
                return ClientOutcome(
                    model_status=ModelStatus.FAILED,
                    model=self._model,
                    error=str(error),
                    attempts=attempts,
                    elapsed_seconds=time.monotonic() - started,
                )
            except TypeSafeError as error:
                attempts.append(
                    RequestAttempt(attempt=attempt_number, status="error", error=str(error))
                )
                return ClientOutcome(
                    model_status=ModelStatus.FAILED,
                    model=self._model,
                    error=str(error),
                    attempts=attempts,
                    elapsed_seconds=time.monotonic() - started,
                )
            attempts.append(RequestAttempt(attempt=attempt_number, status="ok"))
            input_tokens = getattr(response.usage, "input_tokens", None)
            output_tokens = getattr(response.usage, "output_tokens", None)
            self.budget.register_usage(input_tokens, output_tokens)
            request_id = None
            try:
                request_id = response.request_id
            except TypeSafeError:
                request_id = None
            return ClientOutcome(
                model_status=ModelStatus.OK,
                model=getattr(response, "model", self._model),
                request_id=request_id,
                answers=dict(response.answers),
                usage=UsageInfo(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    known=input_tokens is not None,
                ),
                attempts=attempts,
                elapsed_seconds=time.monotonic() - started,
            )
        return ClientOutcome(
            model_status=ModelStatus.FAILED,
            model=self._model,
            error="retry attempts exhausted",
            attempts=attempts,
            elapsed_seconds=time.monotonic() - started,
        )
