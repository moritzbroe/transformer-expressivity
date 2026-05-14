#!/usr/bin/env python3
"""Convert the sudoku-extreme CSV dumps into plain-text puzzle lists."""

import csv
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_DIR / "data"


def clean_csv(csv_path: Path) -> None:
    txt_path = csv_path.with_suffix(".txt")
    count = 0

    with csv_path.open("r", newline="") as src, txt_path.open("w") as dst:
        reader = csv.DictReader(src)
        if not reader.fieldnames or "question" not in reader.fieldnames:
            raise SystemExit(f"{csv_path} is missing the 'question' column")
        for row in reader:
            puzzle = (row.get("question") or "").strip()
            if not puzzle:
                continue
            dst.write(puzzle)
            dst.write("\n")
            count += 1

    if count == 0:
        raise SystemExit(f"No puzzles found in {csv_path}")

    csv_path.unlink()
    txt_rel = txt_path.relative_to(DATA_DIR)
    print(f"Converted {csv_path.name} -> {txt_rel} with {count} puzzles")


if __name__ == "__main__":
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {DATA_DIR}")
    for csv_file in csv_files:
        clean_csv(csv_file)

