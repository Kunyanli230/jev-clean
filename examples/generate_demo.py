"""Generate the fixed-seed demo dataset, ground truth and corruption manifest.

Seed 42 produces 200 unique customer profiles. The dirty file adds explicit
corruption categories (whitespace, null tokens, thousands separators, missing
values, out-of-range ages, two declared date formats, one ambiguous date) and
finally copies 10 rows to demonstrate exact-duplicate removal. Duplicate sources
exclude the ambiguous date row. Ground truth and the corruption manifest are for
the evaluation program only and never enter candidate generation or model state.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
ROW_COUNT = 200
COPY_COUNT = 10

CITIES = ["Berlin", "Hamburg", "Munich", "Cologne", "Frankfurt"]
FIRST_NAMES = [
    "Anna",
    "Ben",
    "Clara",
    "David",
    "Emma",
    "Felix",
    "Greta",
    "Hugo",
    "Ida",
    "Jonas",
    "Klara",
    "Lukas",
    "Mia",
    "Noah",
    "Olga",
    "Paul",
    "Quirin",
    "Rosa",
    "Simon",
    "Tara",
    "Ulf",
    "Vera",
    "Wanda",
    "Xaver",
    "Yara",
    "Zeno",
]
LAST_NAMES = [
    "Albrecht",
    "Bauer",
    "Christ",
    "Dorn",
    "Engel",
    "Fischer",
    "Graf",
    "Hoffmann",
    "Iversen",
    "Jung",
    "Keller",
    "Lang",
    "Meyer",
    "Neumann",
    "Otto",
    "Peters",
    "Quandt",
    "Richter",
    "Schmidt",
    "Thiel",
    "Ulrich",
    "Vogel",
    "Weber",
    "Zimmer",
]

# Explicitly arranged pollution indices, checked for overlap below.
WHITESPACE_NAME_ROWS = list(range(0, 6))
WHITESPACE_CITY_ROWS = list(range(6, 10))
NULL_TOKEN_AGE_ROWS = list(range(10, 14))
NULL_TOKEN_CITY_ROWS = list(range(14, 17))
NULL_TOKEN_INCOME_ROWS = [17]
THOUSANDS_ROWS = list(range(20, 30))
NUMERIC_MISSING_AGE_ROWS = list(range(30, 35))
NUMERIC_MISSING_INCOME_ROWS = list(range(35, 40))
CATEGORICAL_MISSING_ROWS = list(range(40, 45))
OUT_OF_RANGE_AGE_ROWS = list(range(50, 55))
DATE_DMY_ROWS = list(range(60, 65))
DATE_MDY_ROWS = list(range(65, 70))
AMBIGUOUS_DATE_ROW = 70
IQR_HIGH_INCOME_ROW = 80
DUPLICATE_SOURCE_ROWS = [1, 12, 21, 33, 45, 52, 63, 68, 75, 82]


def row_id(index: int) -> str:
    return f"row-{index:06d}"


def build_truth(rng: random.Random) -> list[dict[str, str]]:
    start = date(2018, 1, 1)
    span = (date(2025, 12, 31) - start).days
    rows = []
    for index in range(ROW_COUNT):
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        age = rng.randint(18, 90)
        city = rng.choice(CITIES)
        income = round(rng.uniform(25_000, 180_000), 2)
        signup = start + timedelta(days=rng.randint(0, span))
        if index in set(DATE_DMY_ROWS + DATE_MDY_ROWS) and signup.day <= 12:
            signup = signup.replace(day=signup.day + 12)
        rows.append(
            {
                "customer_id": f"{index + 1:05d}",
                "name": name,
                "age": str(age),
                "city": city,
                "income": f"{income:.2f}",
                "signup_date": signup.isoformat(),
            }
        )
    return rows


def corrupt(
    truth: list[dict[str, str]], rng: random.Random
) -> tuple[list[dict[str, str]], list[dict]]:
    dirty = [dict(row) for row in truth]
    entries: list[dict] = []

    def record(index: int, column: str, corruption_type: str, truth_value: str, recoverable: bool = True) -> None:
        entries.append(
            {
                "row_id": row_id(index),
                "customer_id": truth[index]["customer_id"],
                "column": column,
                "corruption_type": corruption_type,
                "dirty_value": dirty[index][column],
                "truth_value": truth_value,
                "recoverable_by_rule": recoverable,
            }
        )

    for index in WHITESPACE_NAME_ROWS:
        dirty[index]["name"] = f"  {dirty[index]['name']} "
        record(index, "name", "whitespace", truth[index]["name"])
    for index in WHITESPACE_CITY_ROWS:
        dirty[index]["city"] = f" {dirty[index]['city']}  "
        record(index, "city", "whitespace", truth[index]["city"])
    for index in NULL_TOKEN_AGE_ROWS:
        dirty[index]["age"] = "NA"
        record(index, "age", "null_token", truth[index]["age"])
    for index in NULL_TOKEN_CITY_ROWS:
        dirty[index]["city"] = "unknown"
        record(index, "city", "null_token", truth[index]["city"])
    for index in NULL_TOKEN_INCOME_ROWS:
        dirty[index]["income"] = "N/A"
        record(index, "income", "null_token", truth[index]["income"])
    for index in THOUSANDS_ROWS:
        dirty[index]["income"] = f"{float(truth[index]['income']):,.2f}"
        record(index, "income", "numeric_format", truth[index]["income"])
    for index in NUMERIC_MISSING_AGE_ROWS:
        dirty[index]["age"] = ""
        record(index, "age", "numeric_missing", truth[index]["age"])
    for index in NUMERIC_MISSING_INCOME_ROWS:
        dirty[index]["income"] = ""
        record(index, "income", "numeric_missing", truth[index]["income"])
    for index in CATEGORICAL_MISSING_ROWS:
        dirty[index]["city"] = ""
        record(index, "city", "categorical_missing", truth[index]["city"])
    out_of_range_values = [3, 130, 999, 0, 200]
    for offset, index in enumerate(OUT_OF_RANGE_AGE_ROWS):
        dirty[index]["age"] = str(out_of_range_values[offset])
        record(index, "age", "out_of_range", truth[index]["age"])
    for index in DATE_DMY_ROWS:
        source = date.fromisoformat(truth[index]["signup_date"])
        assert source.day > 12, "unique d/m/Y pollution needs day > 12"
        dirty[index]["signup_date"] = source.strftime("%d/%m/%Y")
        record(index, "signup_date", "date_format_dmy", truth[index]["signup_date"])
    for index in DATE_MDY_ROWS:
        source = date.fromisoformat(truth[index]["signup_date"])
        assert source.day > 12, "unique m/d/Y pollution needs day > 12"
        dirty[index]["signup_date"] = source.strftime("%m/%d/%Y")
        record(index, "signup_date", "date_format_mdy", truth[index]["signup_date"])
    ambiguous_truth = date(2024, 4, 3)
    dirty[AMBIGUOUS_DATE_ROW]["signup_date"] = "03/04/2024"
    record(
        AMBIGUOUS_DATE_ROW,
        "signup_date",
        "ambiguous_date",
        ambiguous_truth.isoformat(),
        recoverable=False,
    )
    dirty[IQR_HIGH_INCOME_ROW]["income"] = "950000.00"
    record(
        IQR_HIGH_INCOME_ROW,
        "income",
        "iqr_high_income",
        "950000.00",
        recoverable=False,
    )
    _ = rng
    return dirty, entries


def append_duplicates(dirty: list[dict[str, str]]) -> dict[str, str]:
    sources: dict[str, str] = {}
    assert AMBIGUOUS_DATE_ROW not in DUPLICATE_SOURCE_ROWS
    for offset, source_index in enumerate(DUPLICATE_SOURCE_ROWS):
        copy_index = ROW_COUNT + offset
        dirty.append(dict(dirty[source_index]))
        sources[row_id(copy_index)] = row_id(source_index)
    return sources


def assert_no_overlap() -> None:
    groups = {
        "whitespace_name": WHITESPACE_NAME_ROWS,
        "whitespace_city": WHITESPACE_CITY_ROWS,
        "null_age": NULL_TOKEN_AGE_ROWS,
        "null_city": NULL_TOKEN_CITY_ROWS,
        "null_income": NULL_TOKEN_INCOME_ROWS,
        "thousands": THOUSANDS_ROWS,
        "missing_age": NUMERIC_MISSING_AGE_ROWS,
        "missing_income": NUMERIC_MISSING_INCOME_ROWS,
        "missing_city": CATEGORICAL_MISSING_ROWS,
        "out_of_range": OUT_OF_RANGE_AGE_ROWS,
        "date_dmy": DATE_DMY_ROWS,
        "date_mdy": DATE_MDY_ROWS,
        "ambiguous": [AMBIGUOUS_DATE_ROW],
        "iqr": [IQR_HIGH_INCOME_ROW],
    }
    seen: dict[int, str] = {}
    for name, rows in groups.items():
        for index in rows:
            if index in seen:
                raise AssertionError(f"row {index} polluted by both {seen[index]} and {name}")
            seen[index] = name


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    columns = ["customer_id", "name", "age", "city", "income", "signup_date"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the IDAC demo dataset.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "data"),
        help="Directory for dirty.csv, ground_truth.csv and corruption.json.",
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    assert_no_overlap()
    rng = random.Random(SEED)
    truth = build_truth(rng)
    dirty, entries = corrupt(truth, rng)
    duplicate_sources = append_duplicates(dirty)
    write_csv(output / "dirty.csv", dirty)
    write_csv(output / "ground_truth.csv", truth)
    manifest = {
        "seed": SEED,
        "truth_row_count": len(truth),
        "dirty_row_count": len(dirty),
        "ambiguous_date_row_id": row_id(AMBIGUOUS_DATE_ROW),
        "ambiguous_date_dirty_value": "03/04/2024",
        "ambiguous_date_truth_value": "2024-04-03",
        "iqr_high_income_row_id": row_id(IQR_HIGH_INCOME_ROW),
        "duplicate_sources": duplicate_sources,
        "entries": entries,
    }
    (output / "corruption.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {output / 'dirty.csv'} ({len(dirty)} rows)")
    print(f"wrote {output / 'ground_truth.csv'} ({len(truth)} rows)")
    print(f"wrote {output / 'corruption.json'} ({len(entries)} corruption entries)")


if __name__ == "__main__":
    main()
