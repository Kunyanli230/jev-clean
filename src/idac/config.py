"""YAML configuration loading and validation.

The configuration is a data-authorization declaration: it lists logical column
types, business meaning, null tokens, ranges, date formats, protected columns,
allowed operations and the exact-duplicate comparison columns. Unknown fields
are rejected; the model is never allowed to modify the configuration.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import Operation, fingerprint

ColumnType = Literal["string", "integer", "decimal", "date", "categorical"]

NUMERIC_TYPES = {"integer", "decimal"}
IMPUTABLE_TYPES = {"integer", "decimal", "categorical"}


class ConfigError(ValueError):
    """Raised when the YAML configuration is missing or invalid."""


class ColumnConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ColumnType
    description: str
    allowed_operations: list[Operation] = Field(default_factory=list)
    null_tokens: list[str] = Field(default_factory=list)
    protected: bool = False
    min_value: Decimal | None = None
    max_value: Decimal | None = None
    date_formats: list[str] = Field(default_factory=list)
    allowed_values: list[str] = Field(default_factory=list)
    decimal_places: int | None = None
    decimal_separator: str = "."
    thousands_separator: str | None = None

    @field_validator("min_value", "max_value", mode="before")
    @classmethod
    def _coerce_decimal(cls, value: Any) -> Any:
        if value is None or isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError) as error:
            raise ValueError(f"not a decimal value: {value!r}") from error

    @model_validator(mode="after")
    def _check_column(self) -> ColumnConfig:
        if "" not in self.null_tokens:
            raise ValueError(
                "null_tokens must declare the empty string as a null marker "
                "(CSV exports represent null as an empty field)"
            )
        if len(self.decimal_separator) != 1:
            raise ValueError("decimal_separator must be a single character")
        if self.thousands_separator is not None:
            if len(self.thousands_separator) != 1:
                raise ValueError("thousands_separator must be a single character")
            if self.thousands_separator == self.decimal_separator:
                raise ValueError("thousands_separator must differ from decimal_separator")
        if self.type in NUMERIC_TYPES:
            if self.min_value is not None and self.max_value is not None and self.min_value > self.max_value:
                raise ValueError("min_value must not exceed max_value")
            if self.decimal_places is not None and self.decimal_places < 0:
                raise ValueError("decimal_places must be non-negative")
        if self.type == "date" and not self.date_formats:
            raise ValueError("date columns require at least one date format")
        if self.type == "categorical" and not self.allowed_values:
            raise ValueError("categorical columns require allowed_values")
        if self.type != "date" and self.date_formats:
            raise ValueError("date_formats are only valid for date columns")
        if self.type != "categorical" and self.allowed_values:
            raise ValueError("allowed_values are only valid for categorical columns")
        return self

    def allows(self, operation: Operation) -> bool:
        return operation in self.allowed_operations and not self.protected


class DuplicatesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    compare_columns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_duplicates(self) -> DuplicatesConfig:
        if self.enabled and not self.compare_columns:
            raise ValueError("duplicates.enabled requires compare_columns")
        if not self.enabled and self.compare_columns:
            raise ValueError("duplicates.compare_columns require duplicates.enabled")
        if len(set(self.compare_columns)) != len(self.compare_columns):
            raise ValueError("duplicates.compare_columns must be unique")
        return self


class TableConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_description: str
    record_grain: str
    columns: dict[str, ColumnConfig]
    duplicates: DuplicatesConfig

    @model_validator(mode="after")
    def _check_table(self) -> TableConfig:
        if not self.columns:
            raise ValueError("at least one column configuration is required")
        unknown = [column for column in self.duplicates.compare_columns if column not in self.columns]
        if unknown:
            raise ValueError(f"duplicates.compare_columns not in columns: {unknown}")
        return self

    def column(self, name: str) -> ColumnConfig:
        return self.columns[name]

    def column_type(self, name: str) -> ColumnType:
        return self.columns[name].type

    def logical_schema(self) -> dict[str, str]:
        return {name: column.type for name, column in self.columns.items()}

    def numeric_columns(self) -> list[str]:
        return [name for name, column in self.columns.items() if column.type in NUMERIC_TYPES]

    def imputable_columns(self) -> list[str]:
        return [
            name
            for name, column in self.columns.items()
            if column.type in IMPUTABLE_TYPES and column.allows(Operation.IMPUTE_MISSING)
        ]


def load_config(path: str | Path) -> TableConfig:
    """Load and validate the YAML configuration, rejecting unknown fields."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"configuration file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"configuration is not valid YAML: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    try:
        return TableConfig.model_validate(raw)
    except Exception as error:  # pydantic ValidationError carries the field paths
        raise ConfigError(f"invalid configuration: {error}") from error


def config_hash(config: TableConfig) -> str:
    return fingerprint(config.model_dump(mode="json"))
