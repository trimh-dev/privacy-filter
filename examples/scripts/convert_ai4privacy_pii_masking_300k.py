#!/usr/bin/env python3
"""Convert ai4privacy/pii-masking-300k rows into OPF eval JSONL.

The OPF evaluator expects local records with:

    {"text": "...", "spans": {"LABEL: value": [[start, end]]}, "info": {...}}

Ai4Privacy rows provide source text plus span offsets in either
``privacy_mask`` or ``span_labels``. This script keeps source labels by
default, which is intended for:

    opf eval OUT.jsonl --eval-mode untyped

Use ``--label-map`` to map PII-Masking-300k labels to the native
Privacy Filter categories documented in the OpenAI Privacy Filter model card.
Labels without a corresponding Privacy Filter definition are dropped.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

DATASET_NAME = "ai4privacy/pii-masking-300k"

# Mapping from OpenAI Privacy Filter model card, section 7.2.1,
# "PII-Masking-300k". Keys are normalized by uppercasing and removing
# underscores/hyphens before lookup.
OPF_V2_LABEL_MAP = {
    "ACCOUNT": "account_number",
    "BANKNUM": "account_number",
    "BIC": "account_number",
    "CREDITCARD": "account_number",
    "CRYPTOADDRESS": "account_number",
    "DOCNUM": "account_number",
    "DRIVERLICENSE": "account_number",
    "IBAN": "account_number",
    "IDCARD": "account_number",
    "PASSPORT": "account_number",
    "SOCIALNUMBER": "account_number",
    "TAXNUM": "account_number",
    "BANKMUNICIP": "private_address",
    "BANKPOSTCODE": "private_address",
    "BANKSTREET": "private_address",
    "BUILDING": "private_address",
    "CITY": "private_address",
    "GEOCOORD": "private_address",
    "POSTCODE": "private_address",
    "SECADDRESS": "private_address",
    "STREET": "private_address",
    "CARDEXPIRY": "private_date",
    "DATE": "private_date",
    "DOB": "private_date",
    "BOD": "private_date",
    "EMAIL": "private_email",
    "GIVENNAME1": "private_person",
    "GIVENNAME2": "private_person",
    "LASTNAME1": "private_person",
    "LASTNAME2": "private_person",
    "LASTNAME3": "private_person",
    "TITLE": "private_person",
    "USERNAME": "private_person",
    "TEL": "private_phone",
    "IP": "private_url",
    "OTP": "secret",
    "PASS": "secret",
    "PIN": "secret",
}


def parse_nested(value: Any) -> Any:
    """Parse dataset fields that may be stored as JSON-ish strings."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return value


def iter_spans(row: dict[str, Any]) -> Iterable[tuple[int, int, str, str]]:
    privacy_mask = parse_nested(row.get("privacy_mask"))
    if isinstance(privacy_mask, list):
        for item in privacy_mask:
            if not isinstance(item, dict):
                continue
            start = item.get("start")
            end = item.get("end")
            label = item.get("label")
            value = item.get("value")
            if isinstance(start, int) and isinstance(end, int) and isinstance(label, str):
                yield start, end, label, str(value or row["source_text"][start:end])
        return

    span_labels = parse_nested(row.get("span_labels"))
    if isinstance(span_labels, list):
        text = str(row.get("source_text", ""))
        for item in span_labels:
            if (
                isinstance(item, (list, tuple))
                and len(item) >= 3
                and isinstance(item[0], int)
                and isinstance(item[1], int)
            ):
                start, end, label = item[0], item[1], str(item[2])
                yield start, end, label, text[start:end]


def normalize_label(label: str, label_map: bool) -> str | None:
    if not label_map:
        return label
    normalized = label.upper().replace("_", "").replace("-", "")
    return OPF_V2_LABEL_MAP.get(normalized)


def convert_row(row: dict[str, Any], idx: int, label_map: bool) -> dict[str, Any] | None:
    text = str(row.get("source_text") or "")
    if not text:
        return None

    spans: dict[str, list[list[int]]] = defaultdict(list)
    for start, end, source_label, value in iter_spans(row):
        if not (0 <= start < end <= len(text)):
            continue
        label = normalize_label(source_label, label_map)
        if label is None:
            continue
        key = f"{label}: {value}"
        spans[key].append([start, end])

    return {
        "text": text,
        "spans": dict(spans),
        "info": {
            "id": str(row.get("id") or idx),
            "source": DATASET_NAME,
            "language": row.get("language"),
            "label_map_source": f"{DATASET_NAME}/{row.get('set')}"
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--label-map",
        action="store_true",
        help=(
            "Map source labels to native OPF v2 labels using the model-card mapping. "
            "Omit this flag to keep source labels for opf eval --eval-mode untyped."
        ),
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install dependency first: pip install datasets") from exc

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_NAME, split=args.split)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(dataset):
            if args.max_examples is not None and written >= args.max_examples:
                break
            record = convert_row(dict(row), idx, args.label_map)
            if record is None:
                continue
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            written += 1

    print(f"Wrote {written} records to {output_path}")


if __name__ == "__main__":
    main()
