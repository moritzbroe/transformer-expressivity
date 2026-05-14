import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from sudoku_reasoning.tokenizer import SudokuTokenizer


Puzzle = Sequence[int]


def parse_targets(targets_str: str | None, logspace_args: List[str] | None) -> List[int] | None:
    """Parse target CoT lengths from command line arguments.

    Args:
        targets_str: Comma-separated list of integers, e.g. "1024,2048,4096"
        logspace_args: Three values [start, end, count] for log-spaced targets

    Returns:
        List of target integers, or None if no targets specified.
    """
    if targets_str and logspace_args:
        raise ValueError("Cannot specify both --targets and --targets-logspace")

    if targets_str:
        return [int(x.strip()) for x in targets_str.split(",")]

    if logspace_args:
        start, end, n = int(logspace_args[0]), int(logspace_args[1]), int(logspace_args[2])
        if n < 2:
            raise ValueError("--targets-logspace count must be at least 2")
        if start <= 0 or end <= 0:
            raise ValueError("--targets-logspace start and end must be positive")
        if start >= end:
            raise ValueError("--targets-logspace start must be less than end")
        # geomspace: start * (end/start)^(i/(n-1)) for i = 0..n-1
        ratio = end / start
        targets = [int(round(start * (ratio ** (i / (n - 1))))) for i in range(n)]
        return sorted(set(targets))  # remove duplicates from rounding

    return None


def sample_by_targets(
    puzzles: List[Tuple[int, ...]],
    cot_lengths: Dict[str, int],
    targets: List[int],
    per_target: int,
) -> Tuple[List[Tuple[int, Tuple[int, ...], int]], List[int]]:
    """Select per_target puzzles closest to each target CoT length.

    Args:
        puzzles: List of puzzles
        cot_lengths: Dict mapping puzzle string to CoT length
        targets: List of target CoT lengths
        per_target: Number of puzzles to select per target

    Returns:
        Tuple of (samples, target_for_each_sample) where samples is
        List of (row_idx, puzzle, cot_length) tuples.
    """
    items = [(idx, p, cot_lengths["".join(str(c) for c in p)]) for idx, p in enumerate(puzzles)]

    samples = []
    sample_targets = []
    for target in targets:
        ranked = sorted(items, key=lambda x: abs(x[2] - target))
        for item in ranked[:per_target]:
            samples.append(item)
            sample_targets.append(target)

    return samples, sample_targets


Triple = Tuple[int, int, int]

GRID_SIZE = 9
BOX_SIZE = 3
CELL_COUNT = GRID_SIZE * GRID_SIZE


def stream_puzzles(path: str):
    """Yield puzzles from ``path`` one at a time without storing the full corpus."""

    src = Path(path)

    def _generator():
        found = False
        with src.open("r") as fh:
            for line_no, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line:
                    continue
                values: List[int] = []
                for ch in line:
                    if ch in ("0", "."):
                        values.append(0)
                    elif ch.isdigit():
                        values.append(int(ch))
                    else:
                        raise ValueError(f"{src}:{line_no}: invalid character '{ch}'")
                if len(values) != CELL_COUNT:
                    raise ValueError(
                        f"{src}:{line_no}: expected {CELL_COUNT} cells, got {len(values)}"
                    )
                if any(v < 0 or v > GRID_SIZE for v in values):
                    raise ValueError(f"{src}:{line_no}: value out of range 0..{GRID_SIZE}")
                puzzle = tuple(values)
                found = True
                yield puzzle
        if not found:
            raise ValueError(f"No puzzles found in {path}")

    return _generator()


def load_puzzles(path: str) -> List[Tuple[int, ...]]:
    return list(stream_puzzles(path))


def sample_puzzles(path: str, count: int, seed: int | None = None):
    puzzles = load_puzzles(path)
    if count > len(puzzles):
        raise ValueError("Requested more samples than available puzzles")
    rng = random.Random(seed) if seed is not None else random
    indices = rng.sample(range(len(puzzles)), count)
    return [(idx, puzzles[idx]) for idx in indices]


def build_prompt_tokens(puzzle: Puzzle, tokenizer: SudokuTokenizer) -> List[object]:
    if len(puzzle) != CELL_COUNT:
        raise ValueError(f"Puzzle length {len(puzzle)} does not match {CELL_COUNT} cells")
    tokens = [tokenizer.input_token]
    for idx, value in enumerate(puzzle):
        if value == 0:
            continue
        row, col = divmod(idx, GRID_SIZE)
        tokens.extend([(row + 1, col + 1), value])
    tokens.append(tokenizer.input_end_token)
    return tokens


def extract_board_from_output(triples: Iterable[Triple]) -> Tuple[int, ...] | None:
    """Extract a board from 81 ordered (row, col, value) triples.

    Expects triples in row-major order: (1,1), (1,2), ..., (9,9).
    Returns None if malformed.
    """
    board = []
    for expected_idx, (r, c, value) in enumerate(triples):
        if not (1 <= r <= GRID_SIZE and 1 <= c <= GRID_SIZE and 1 <= value <= GRID_SIZE):
            return None
        idx = (r - 1) * GRID_SIZE + (c - 1)
        if idx != expected_idx:
            return None
        board.append(value)
    if len(board) != CELL_COUNT:
        return None
    return tuple(board)


def is_valid_solution(puzzle: Puzzle, candidate: Puzzle) -> bool:
    if len(candidate) != CELL_COUNT or len(puzzle) != CELL_COUNT:
        return False
    if any(value < 1 or value > GRID_SIZE for value in candidate):
        return False
    for idx, given in enumerate(puzzle):
        if given != 0 and candidate[idx] != given:
            return False

    digits = set(range(1, GRID_SIZE + 1))

    for row in range(GRID_SIZE):
        row_vals = candidate[row * GRID_SIZE : (row + 1) * GRID_SIZE]
        if set(row_vals) != digits:
            return False
    for col in range(GRID_SIZE):
        col_vals = [candidate[row * GRID_SIZE + col] for row in range(GRID_SIZE)]
        if set(col_vals) != digits:
            return False
    for br in range(0, GRID_SIZE, BOX_SIZE):
        for bc in range(0, GRID_SIZE, BOX_SIZE):
            block = [
                candidate[(br + r) * GRID_SIZE + (bc + c)]
                for r in range(BOX_SIZE)
                for c in range(BOX_SIZE)
            ]
            if set(block) != digits:
                return False
    return True


__all__ = [
    "Puzzle",
    "Triple",
    "build_prompt_tokens",
    "extract_board_from_output",
    "is_valid_solution",
    "load_puzzles",
    "parse_targets",
    "sample_by_targets",
    "sample_puzzles",
    "stream_puzzles",
]
