#!/usr/bin/env python3
"""Precompute CoT lengths for all puzzles using multiprocessing."""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Tuple

from sudoku_reasoning.puzzle_utils import stream_puzzles
from sudoku_reasoning.solver import solve
from sudoku_reasoning.tokenizer import SudokuTokenizer
from sudoku_reasoning.trace_utils import build_cot_tokens


_WORKER_SETTINGS: Dict = {}


def _init_worker(max_steps: int) -> None:
    _WORKER_SETTINGS["max_steps"] = max_steps
    _WORKER_SETTINGS["tokenizer"] = SudokuTokenizer()


def _compute_cot_length(task: Tuple[int, Tuple[int, ...]]) -> Tuple[int, str, int]:
    idx, puzzle = task
    max_steps = _WORKER_SETTINGS["max_steps"]
    tokenizer = _WORKER_SETTINGS["tokenizer"]

    inputs, trace, outputs, step_count, limit_reached = solve(
        list(puzzle),
        max_steps=max_steps,
        deterministic=True,
    )

    cot_len = len(build_cot_tokens(inputs, trace, outputs, tokenizer))
    puzzle_str = "".join(str(c) for c in puzzle)

    return idx, puzzle_str, cot_len


def _load_puzzles(path: Path) -> List[Tuple[int, ...]]:
    return list(stream_puzzles(str(path)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute CoT lengths for puzzles")
    parser.add_argument(
        "--input",
        default="data/test.txt",
        help="Input puzzle file (default: data/test.txt)",
    )
    parser.add_argument(
        "--output",
        default="data/test_cot_lengths.json",
        help="Output JSON file for CoT lengths (default: data/test_cot_lengths.json)",
    )
    parser.add_argument(
        "--max-solver-steps",
        type=int,
        default=10_000_000,
        help="Maximum solver steps (default: 10000000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes (default: cpu count)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"{input_path} does not exist")

    puzzles = _load_puzzles(input_path)
    total = len(puzzles)
    if total == 0:
        raise SystemExit(f"No puzzles found in {input_path}")

    print(f"Computing CoT lengths for {total} puzzles using {args.workers} workers...")

    tasks = [(idx, puzzle) for idx, puzzle in enumerate(puzzles)]
    results: Dict[str, int] = {}
    processed = 0

    with mp.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(args.max_solver_steps,),
    ) as pool:
        for idx, puzzle_str, cot_len in pool.imap(_compute_cot_length, tasks, chunksize=100):
            results[puzzle_str] = cot_len
            processed += 1
            if processed % 1000 == 0 or processed == total:
                print(f"Processed {processed}/{total}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results))

    print(f"Wrote {len(results)} entries to {output_path}")


if __name__ == "__main__":
    main()
