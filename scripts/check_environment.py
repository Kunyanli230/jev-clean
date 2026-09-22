"""Verify local development dependencies without contacting the model API."""

import sys
from importlib import import_module
from importlib.metadata import version

from typesafe_sdk import Choice, Noul, RetryPolicy, Score, TypeSafeClient


def main() -> None:
    assert sys.version_info[:2] == (3, 11), sys.version
    for module, package in (
        ("pandas", "pandas"),
        ("pydantic", "pydantic"),
        ("yaml", "PyYAML"),
        ("typer", "typer"),
        ("pytest", "pytest"),
        ("typesafe_sdk", "typesafe-sdk"),
    ):
        import_module(module)
        print(f"{package}: {version(package)}")

    Choice(instructions="Which action fits?", criteria={"apply": None, "keep": None})
    Noul(instructions="Does the policy support the candidate?")
    Score(instructions="How risky is the candidate?", criteria=["Low", "Moderate", "High"])
    RetryPolicy(max_retries=0)
    assert callable(TypeSafeClient.system_one)
    print("PASS: Python, imports and SDK question construction. No API requests made.")


if __name__ == "__main__":
    main()
