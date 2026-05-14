#!/usr/bin/env python3
"""Precompute Sudoku training segments and store them as a single Arrow dataset.

Modes:
  - scot: segmented traces with summary tokens.
  - cot: full (unsegmented) traces.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from sudoku_reasoning.puzzle_utils import stream_puzzles
from sudoku_reasoning.solver import solve
from sudoku_reasoning.tokenizer import SudokuTokenizer
from sudoku_reasoning.trace_utils import build_cot_segment, to_tokens


def _build_scot_segments(
    inputs: Sequence[Tuple[Tuple[int, int], int]],
    trace: Sequence[Tuple[Tuple[int, int], Iterable[int]]],
    outputs: Sequence[Tuple[Tuple[int, int], int]] | None,
    tokenizer: SudokuTokenizer,
    segment_trace_tokens: int | None,
) -> List[dict]:
    if outputs is None:
        return []

    input_tokens = [tokenizer.input_token] + to_tokens(inputs, tokenizer) + [tokenizer.input_end_token]
    trace_lines = [to_tokens([line], tokenizer) for line in trace]  # each line already flat
    trace_tokens = [tok for line in trace_lines for tok in line]
    output_tokens = [tokenizer.output_token] + to_tokens(outputs, tokenizer) + [tokenizer.output_end_token]

    full_tokens = input_tokens + trace_tokens + output_tokens
    loss_mask = [0] * len(input_tokens) + [1] * (len(trace_tokens) + len(output_tokens))

    if segment_trace_tokens is None or segment_trace_tokens <= 0:
        return [{"input_ids": tokenizer.tokens_to_ids(full_tokens), "loss_mask": loss_mask}]

    segment_trace_tokens = max(1, segment_trace_tokens)
    stack: List[List[object]] = []
    tokens_since_summary = 0
    trace_prefix: List[object] = []
    summary_entries: List[dict] = []  # each: {"summary": [...], "after": [...]}
    current_after: List[object] | None = None
    prompt_tokens: List[object] | None = None

    def maybe_insert_summary():
        nonlocal tokens_since_summary, current_after, prompt_tokens
        if tokens_since_summary < segment_trace_tokens or not stack:
            return
        snapshot = [tokenizer.summary_token]
        snapshot.extend(input_tokens[1:-1])
        for saved_line in stack:
            snapshot.extend(saved_line)
        snapshot.append(tokenizer.summary_end_token)
        if prompt_tokens is None:
            prompt_tokens = input_tokens + trace_prefix + snapshot
        summary_entries.append({"summary": snapshot, "after": []})
        current_after = summary_entries[-1]["after"]
        tokens_since_summary = 0

    for line in trace_lines:
        maybe_insert_summary()
        tokens_since_summary += len(line)
        if summary_entries:
            if current_after is None:
                current_after = summary_entries[-1]["after"]
            current_after.extend(line)
        else:
            trace_prefix.extend(line)
        coord = line[0]
        for idx, existing in enumerate(stack):
            if existing[0] == coord:
                stack[idx] = line
                del stack[idx + 1 :]
                break
        else:
            stack.append(line)
    # Close with final segments.
    if not summary_entries:
        return [{"input_ids": tokenizer.tokens_to_ids(full_tokens), "loss_mask": loss_mask}]

    summary_segments: List[dict] = []
    prompt_tokens = prompt_tokens or (input_tokens + trace_prefix + summary_entries[0]["summary"])
    prompt_mask = [0] * len(input_tokens) + [1] * (len(prompt_tokens) - len(input_tokens))
    prompt_seg = {"input_ids": tokenizer.tokens_to_ids(prompt_tokens), "loss_mask": prompt_mask}

    for idx in range(len(summary_entries) - 1):
        cur = summary_entries[idx]
        nxt = summary_entries[idx + 1]
        seg_tokens = cur["summary"] + cur["after"] + nxt["summary"]
        zero_len = len(cur["summary"])
        seg_mask = [0] * zero_len + [1] * (len(seg_tokens) - zero_len)
        summary_segments.append(
            {"input_ids": tokenizer.tokens_to_ids(seg_tokens), "loss_mask": seg_mask}
        )

    last = summary_entries[-1]
    tail_tokens = last["summary"] + last["after"] + output_tokens
    zero_len = len(last["summary"])
    tail_mask = [0] * zero_len + [1] * (len(tail_tokens) - zero_len)
    tail_seg = {"input_ids": tokenizer.tokens_to_ids(tail_tokens), "loss_mask": tail_mask}

    return [prompt_seg] + summary_segments + [tail_seg]


Puzzle = Sequence[int]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute Sudoku training data")
    parser.add_argument(
        "--mode",
        choices=["cot", "scot"],
        default="scot",
        help="Training data style: cot (full traces) or scot (segmented with summary tokens)",
    )
    parser.add_argument("--input", default="data/train.txt", help="Text file with puzzles")
    parser.add_argument(
        "--output",
        default="data/train_data",
        help="Directory where the Arrow dataset will be written",
    )
    parser.add_argument(
        "--segment-trace-tokens",
        type=int,
        default=512,
        help="(scot) Insert summaries after approximately this many trace tokens; set <=0 for full traces",
    )
    parser.add_argument(
        "--max-cot-length",
        type=int,
        default=1 << 14,
        help="Skip puzzles whose full COT token sequence (input + trace + output) is longer than this",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use randomized solver decisions when generating traces",
    )
    return parser.parse_args()


def main() -> None:
    from datasets import Dataset, Features, Sequence as HFSequence, Value

    args = parse_args()
    if args.max_cot_length is None or args.max_cot_length <= 0:
        raise SystemExit("--max-cot-length must be > 0")

    segment_trace_tokens = args.segment_trace_tokens
    if args.mode == "cot":
        segment_trace_tokens = None

    output_dir = Path(args.output)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory {output_dir} is not empty.")
    output_dir.mkdir(parents=True, exist_ok=True)

    preview = stream_puzzles(args.input)
    try:
        first_puzzle = next(preview)
    except StopIteration:
        raise SystemExit("No puzzles found to process")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        preview.close()

    tokenizer = SudokuTokenizer()
    max_solver_steps = max(int(args.max_cot_length), 1000)

    def puzzle_generator():
        puzzles = stream_puzzles(args.input)
        for puzzle in puzzles:
            yield {"puzzle": list(puzzle)}

    raw = Dataset.from_generator(
        puzzle_generator,
        features=Features({"puzzle": HFSequence(Value("int16"))}),
        keep_in_memory=False,
    )

    features = Features(
        {
            "input_ids": HFSequence(Value("int32")),
            "loss_mask": HFSequence(Value("int8")),
            "length": Value("int32"),
            "cot_length": Value("int32"),
        }
    )

    def expand_batch(batch):
        ids_list: List[List[int]] = []
        mask_list: List[List[int]] = []
        lengths_list: List[int] = []
        cot_lengths_list: List[int] = []
        for puzzle in batch["puzzle"]:
            inputs, trace, outputs, _, limit_reached = solve(
                list(puzzle),
                max_steps=max_solver_steps,
                deterministic=not args.random,
            )
            if limit_reached or outputs is None:
                continue
            cot = build_cot_segment(inputs, trace, outputs, tokenizer=tokenizer)
            cot_length = len(cot["input_ids"])
            if cot_length > args.max_cot_length:
                continue
            if args.mode == "cot":
                ids_list.append(cot["input_ids"])
                mask_list.append(cot["loss_mask"])
                lengths_list.append(cot_length)
                cot_lengths_list.append(cot_length)
                continue
            for seg in _build_scot_segments(
                inputs,
                trace,
                outputs,
                tokenizer=tokenizer,
                segment_trace_tokens=segment_trace_tokens,
            ):
                ids_list.append(seg["input_ids"])
                mask_list.append(seg["loss_mask"])
                lengths_list.append(len(seg["input_ids"]))
                cot_lengths_list.append(cot_length)
        return {
            "input_ids": ids_list,
            "loss_mask": mask_list,
            "length": lengths_list,
            "cot_length": cot_lengths_list,
        }

    num_proc = max(1, int(args.workers))
    map_kwargs = dict(
        batched=True,
        batch_size=1,
        remove_columns=["puzzle"],
        features=features,
        writer_batch_size=1000,
    )
    if num_proc > 1:
        map_kwargs["num_proc"] = num_proc

    dataset = raw.map(expand_batch, **map_kwargs)

    if len(dataset) == 0:
        raise SystemExit("No training segments were generated; aborting.")

    dataset.save_to_disk(str(output_dir))

    metadata = {
        "pad_id": tokenizer.pad_id,
        "num_segments": len(dataset),
        "mode": args.mode,
        "segment_trace_tokens": segment_trace_tokens,
        "max_cot_length": args.max_cot_length,
        "max_solver_steps": max_solver_steps,
        "random": bool(args.random),
    }
    with (output_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Wrote {len(dataset)} segments to {output_dir}")


if __name__ == "__main__":
    main()
