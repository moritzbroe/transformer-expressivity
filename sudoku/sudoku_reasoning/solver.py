"""Sudoku solver instrumented to produce MRV decision traces."""
import random


GRID_SIZE = 9
BOX_SIZE = 3
CELL_COUNT = GRID_SIZE * GRID_SIZE


def _prepare_puzzle(puzzle):
    if not isinstance(puzzle, (list, tuple)):
        raise TypeError("Puzzle must be provided as a flat list or tuple of ints")
    length = len(puzzle)
    if length == 0:
        raise ValueError("Puzzle cannot be empty")
    if length != CELL_COUNT:
        raise ValueError(f"Puzzle length must be {CELL_COUNT} for 9x9 Sudoku")
    board = []
    for idx, value in enumerate(puzzle):
        if not isinstance(value, int):
            raise TypeError(f"Puzzle entries must be ints (index {idx})")
        if value < 0 or value > GRID_SIZE:
            raise ValueError(f"Puzzle entries must be in 0..{GRID_SIZE} (index {idx})")
        board.append(value)
    return GRID_SIZE, BOX_SIZE, board


def solve(puzzle, max_steps=None, deterministic=False):
    """Solve ``puzzle`` (as a flat list of ints) while recording MRV decisions.

    Returns (inputs, trace, outputs, step_count, limit_reached) where:
      - inputs: the givens as [((row, col), digit)]
      - trace:  a list of ((row, col), options) decisions describing the search
      - outputs: all 81 cells as [((row, col), digit)] if solved, else None
      - step_count: number of assignment attempts performed by the solver
      - limit_reached: True if the run terminated because ``max_steps`` was hit
    """

    size, box_size, board = _prepare_puzzle(puzzle)
    board_cells = size * size

    original_values = board[:]
    row_digits = [set() for _ in range(size)]
    col_digits = [set() for _ in range(size)]
    box_digits = [set() for _ in range(size)]

    for idx, value in enumerate(board):
        if value == 0:
            continue
        row, col = divmod(idx, size)
        box = (row // box_size) * box_size + (col // box_size)
        if (
            value in row_digits[row]
            or value in col_digits[col]
            or value in box_digits[box]
        ):
            raise ValueError("Puzzle contains conflicting givens")
        row_digits[row].add(value)
        col_digits[col].add(value)
        box_digits[box].add(value)

    trace = []
    rng = random.Random()
    steps = 0
    limit_reached = False

    def select_cell():
        best_cells = []
        min_options = size + 1
        for idx in range(board_cells):
            if board[idx] > 0:
                continue
            row, col = divmod(idx, size)
            box = (row // box_size) * box_size + (col // box_size)
            candidates = [
                d
                for d in range(1, size + 1)
                if d not in row_digits[row]
                and d not in col_digits[col]
                and d not in box_digits[box]
            ]
            count = len(candidates)
            if count == 0:
                return (idx, row, col), []
            if count < min_options:
                min_options = count
                best_cells = [(idx, row, col, candidates)]
            elif count == min_options:
                best_cells.append((idx, row, col, candidates))
        if not best_cells:
            return (None, None, None), []
        if deterministic:
            idx, row, col, candidates = best_cells[0]
            options = candidates[:]
        else:
            idx, row, col, candidates = rng.choice(best_cells)
            options = candidates[:]
            rng.shuffle(options)
        return (idx, row, col), options

    def backtrack():
        nonlocal steps, limit_reached
        cell, options = select_cell()
        idx, row, col = cell
        if idx is None:
            return True
        coord = (row + 1, col + 1)
        trace.append((coord, options[:]))
        if not options:
            return False
        box = (row // box_size) * box_size + (col // box_size)
        for i, digit in enumerate(options):
            if max_steps is not None and steps >= max_steps:
                limit_reached = True
                return False
            steps += 1
            board[idx] = digit
            row_digits[row].add(digit)
            col_digits[col].add(digit)
            box_digits[box].add(digit)
            if backtrack():
                return True
            board[idx] = 0
            row_digits[row].remove(digit)
            col_digits[col].remove(digit)
            box_digits[box].remove(digit)
            if limit_reached:
                return False
            remaining = options[i + 1 :]
            if remaining:
                trace.append((coord, remaining[:]))
        return False

    solved = backtrack()

    inputs = []
    outputs = [] if solved and not limit_reached else None

    for idx in range(size * size):
        row, col = divmod(idx, size)
        coord = (row + 1, col + 1)
        if original_values[idx] > 0:
            inputs.append((coord, original_values[idx]))
        if outputs is not None:
            outputs.append((coord, board[idx]))

    return inputs, trace, outputs, steps, limit_reached


__all__ = ["solve"]
