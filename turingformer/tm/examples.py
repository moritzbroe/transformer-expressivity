from __future__ import annotations

from itertools import product
from typing import Dict, Tuple, cast

from tm.turing import MultiTapeTuringMachine


def binary_addition_machine() -> MultiTapeTuringMachine:
    transitions = {
        # write start symbol on second tape
        ("q0", ("1", "_")): ("q0.5", ("1", "s"), ("S", "R")),
        ("q0", ("0", "_")): ("q0.5", ("0", "s"), ("S", "R")),
        # move first number to second tape, write start symbol on first tape
        ("q0.5", ("1", "_")): ("q1", ("s", "1"), ("R", "R")),
        ("q0.5", ("0", "_")): ("q1", ("s", "0"), ("R", "R")),
        ("q1", ("0", "_")): ("q1", ("_", "0"), ("R", "R")),
        ("q1", ("1", "_")): ("q1", ("_", "1"), ("R", "R")),
        ("q1", ("+", "_")): ("q2", ("_", "_"), ("R", "S")),
        # move the head of the first tape to the right
        ("q2", ("0", "_")): ("q2", ("0", "_"), ("R", "S")),
        ("q2", ("1", "_")): ("q2", ("1", "_"), ("R", "S")),
        ("q2", ("_", "_")): ("q3", ("_", "_"), ("L", "L")),
        # add number on second tape to number on first tape, qc is carry state, q3 is no carry state
        # when we are not in carry state, we take the sum mod 2 as the written bit, and the sum // 2 as the carry bit
        ("q3", ("0", "0")): ("q3", ("0", "_"), ("L", "L")),
        ("q3", ("0", "s")): ("q3", ("0", "s"), ("L", "S")),
        ("q3", ("_", "0")): ("q3", ("0", "_"), ("L", "L")),
        ("q3", ("1", "0")): ("q3", ("1", "_"), ("L", "L")),
        ("q3", ("1", "s")): ("q3", ("1", "s"), ("L", "S")),
        ("q3", ("0", "1")): ("q3", ("1", "_"), ("L", "L")),
        ("q3", ("_", "1")): ("q3", ("1", "_"), ("L", "L")),
        ("q3", ("1", "1")): ("qc", ("0", "_"), ("L", "L")),
        # in carry state, add carry bit as well
        ("qc", ("0", "0")): ("q3", ("1", "_"), ("L", "L")),
        ("qc", ("0", "s")): ("q3", ("1", "s"), ("L", "S")),
        ("qc", ("_", "0")): ("q3", ("1", "_"), ("L", "L")),
        ("qc", ("_", "s")): ("q3", ("1", "s"), ("L", "S")),
        ("qc", ("_", "1")): ("qc", ("0", "_"), ("L", "L")),
        ("qc", ("1", "0")): ("qc", ("0", "_"), ("L", "L")),
        ("qc", ("0", "1")): ("qc", ("0", "_"), ("L", "L")),
        ("qc", ("1", "1")): ("qc", ("1", "_"), ("L", "L")),
        ("qc", ("1", "s")): ("qc", ("0", "s"), ("L", "S")),
        # handle ends: if we still have a carry we need to write it, otherwise we are done if we hit start symbol on tape 2 and blank on tape 1.
        ("qc", ("_", "s")): ("q4", ("1", "s"), ("S", "R")),
        ("q3", ("_", "s")): ("q4", ("_", "s"), ("R", "R")),
        # now we have the result but shifted to the left. write it on tape 2, then move it back to tape 1.
        ("q4", ("0", "_")): ("q4", ("_", "0"), ("R", "R")),
        ("q4", ("1", "_")): ("q4", ("_", "1"), ("R", "R")),
        ("q4", ("_", "_")): ("q5", ("_", "_"), ("L", "S")),
        # move all the way to the left
        ("q5", ("_", "_")): ("q5", ("_", "_"), ("L", "L")),
        ("q5", ("_", "0")): ("q5", ("_", "0"), ("L", "L")),
        ("q5", ("_", "1")): ("q5", ("_", "1"), ("L", "L")),
        ("q5", ("_", "s")): ("q5", ("_", "s"), ("L", "L")),
        ("q5", ("s", "_")): ("q5", ("s", "_"), ("L", "L")),
        ("q5", ("s", "0")): ("q5", ("s", "0"), ("L", "L")),
        ("q5", ("s", "1")): ("q5", ("s", "1"), ("L", "L")),
        ("q5", ("s", "s")): ("q6", ("s", "s"), ("S", "R")),
        # copy tape 2 to tape 1
        ("q6", ("_", "0")): ("q6", ("0", "_"), ("R", "R")),
        ("q6", ("_", "1")): ("q6", ("1", "_"), ("R", "R")),
        ("q6", ("s", "0")): ("q6", ("0", "_"), ("R", "R")),
        ("q6", ("s", "1")): ("q6", ("1", "_"), ("R", "R")),
        ("q6", ("_", "_")): ("halt", ("_", "_"), ("S", "S")),
    }

    return MultiTapeTuringMachine(
        num_tapes=2,
        transitions=transitions,
        initial_state="q0",
        halting_state="halt",
        input_vocabulary=["0", "1", "+"],
        blank="_",
    )


def binary_multiplication_machine() -> MultiTapeTuringMachine:
    transitions: Dict[
        Tuple[str, Tuple[str, str, str]],
        Tuple[str, Tuple[str, str, str], Tuple[str, str, str]],
    ] = {}
    symbols = ("0", "1", "_", "s", "*")

    def expand(pattern: Tuple[str, str, str]):
        choices = []
        for entry in pattern:
            if entry == "?":
                choices.append(symbols)
            else:
                choices.append((entry,))
        for combo in product(*choices):
            yield combo

    def add_rule(
        state: str,
        read_pattern: Tuple[str, str, str],
        next_state: str,
        write_pattern: Tuple[str, str, str],
        move_pattern: Tuple[str, str, str],
    ) -> None:
        for read in expand(read_pattern):
            write = []
            for idx, spec in enumerate(write_pattern):
                if spec == "=":
                    write.append(read[idx])
                else:
                    write.append(spec)
            key = (state, read)
            if key in transitions:
                raise ValueError(f"duplicate transition defined for {key}")
            transitions[key] = (next_state, (write[0], write[1], write[2]), move_pattern)

    def add_insert_subroutine(prefix: str, tape_index: int, return_state: str) -> None:
        def pattern(symbol: str) -> Tuple[str, str, str]:
            arr = ["?", "?", "?"]
            arr[tape_index] = symbol
            return cast(Tuple[str, str, str], tuple(arr))

        def write(symbol: str) -> Tuple[str, str, str]:
            arr = ["=", "=", "="]
            arr[tape_index] = symbol
            return cast(Tuple[str, str, str], tuple(arr))

        def move(direction: str) -> Tuple[str, str, str]:
            arr = ["S", "S", "S"]
            arr[tape_index] = direction
            return cast(Tuple[str, str, str], tuple(arr))

        add_rule(f"{prefix}_start_0", pattern("s"), f"{prefix}_loop_0", write("s"), move("R"))
        add_rule(f"{prefix}_start_1", pattern("s"), f"{prefix}_loop_1", write("s"), move("R"))

        add_rule(f"{prefix}_loop_0", pattern("0"), f"{prefix}_loop_0", write("0"), move("R"))
        add_rule(f"{prefix}_loop_0", pattern("1"), f"{prefix}_loop_1", write("0"), move("R"))
        add_rule(f"{prefix}_loop_0", pattern("_"), f"{prefix}_return", write("0"), move("L"))

        add_rule(f"{prefix}_loop_1", pattern("0"), f"{prefix}_loop_0", write("1"), move("R"))
        add_rule(f"{prefix}_loop_1", pattern("1"), f"{prefix}_loop_1", write("1"), move("R"))
        add_rule(f"{prefix}_loop_1", pattern("_"), f"{prefix}_return", write("1"), move("L"))

        add_rule(f"{prefix}_return", pattern("0"), f"{prefix}_return", ("=", "=", "="), move("L"))
        add_rule(f"{prefix}_return", pattern("1"), f"{prefix}_return", ("=", "=", "="), move("L"))
        add_rule(f"{prefix}_return", pattern("_"), f"{prefix}_return", ("=", "=", "="), move("L"))
        add_rule(f"{prefix}_return", pattern("s"), return_state, ("=", "=", "="), ("S", "S", "S"))

    add_insert_subroutine("t3_ins_copyA", 2, "copy_A_read")
    add_insert_subroutine("t2_ins_copyB", 1, "copy_B_read")
    add_insert_subroutine("t2_ins_final", 1, "final_copy_to_t2")

    add_rule("init", ("0", "_", "_"), "t3_ins_copyA_start_0", ("s", "s", "s"), ("R", "S", "S"))
    add_rule("init", ("1", "_", "_"), "t3_ins_copyA_start_1", ("s", "s", "s"), ("R", "S", "S"))

    add_rule("copy_A_read", ("0", "s", "s"), "t3_ins_copyA_start_0", ("_", "=", "="), ("R", "S", "S"))
    add_rule("copy_A_read", ("1", "s", "s"), "t3_ins_copyA_start_1", ("_", "=", "="), ("R", "S", "S"))
    add_rule("copy_A_read", ("*", "s", "s"), "copy_B_read", ("_", "=", "="), ("R", "S", "S"))

    add_rule("copy_B_read", ("0", "s", "s"), "t2_ins_copyB_start_0", ("_", "=", "="), ("R", "S", "S"))
    add_rule("copy_B_read", ("1", "s", "s"), "t2_ins_copyB_start_1", ("_", "=", "="), ("R", "S", "S"))
    add_rule("copy_B_read", ("_", "s", "s"), "after_copy_move_left", ("=", "=", "="), ("L", "S", "S"))

    add_rule("after_copy_move_left", ("_", "s", "s"), "after_copy_move_left", ("=", "=", "="), ("L", "S", "S"))
    add_rule("after_copy_move_left", ("s", "s", "s"), "loop_check", ("=", "=", "="), ("S", "S", "S"))

    add_rule("loop_check", ("s", "s", "s"), "loop_check_scan", ("=", "=", "="), ("S", "R", "S"))

    add_rule("loop_check_scan", ("s", "0", "s"), "loop_check_scan", ("=", "=", "="), ("S", "R", "S"))
    add_rule("loop_check_scan", ("s", "1", "s"), "loop_back_to_start_after_check", ("=", "=", "="), ("S", "L", "S"))
    add_rule("loop_check_scan", ("s", "_", "s"), "loop_back_to_start_zero", ("=", "=", "="), ("S", "L", "S"))

    add_rule(
        "loop_back_to_start_after_check",
        ("s", "0", "s"),
        "loop_back_to_start_after_check",
        ("=", "=", "="),
        ("S", "L", "S"),
    )
    add_rule(
        "loop_back_to_start_after_check",
        ("s", "1", "s"),
        "loop_back_to_start_after_check",
        ("=", "=", "="),
        ("S", "L", "S"),
    )
    add_rule(
        "loop_back_to_start_after_check",
        ("s", "_", "s"),
        "loop_back_to_start_after_check",
        ("=", "=", "="),
        ("S", "L", "S"),
    )
    add_rule(
        "loop_back_to_start_after_check",
        ("s", "s", "s"),
        "add_prepare_t1",
        ("=", "=", "="),
        ("S", "S", "S"),
    )

    add_rule("loop_back_to_start_zero", ("s", "0", "s"), "loop_back_to_start_zero", ("=", "=", "="), ("S", "L", "S"))
    add_rule("loop_back_to_start_zero", ("s", "1", "s"), "loop_back_to_start_zero", ("=", "=", "="), ("S", "L", "S"))
    add_rule("loop_back_to_start_zero", ("s", "_", "s"), "loop_back_to_start_zero", ("=", "=", "="), ("S", "L", "S"))
    add_rule("loop_back_to_start_zero", ("s", "s", "s"), "final_copy_to_t2_prepare", ("=", "=", "="), ("S", "S", "S"))

    add_rule("add_prepare_t1", ("s", "s", "s"), "add_prepare_t3", ("=", "=", "="), ("R", "S", "S"))
    add_rule("add_prepare_t3", ("?", "s", "s"), "add_loop_no_carry", ("=", "=", "="), ("S", "S", "R"))

    digits = ("0", "1", "_")
    for a in digits:
        for b in digits:
            if a == "_" and b == "_":
                continue
            total = (1 if a == "1" else 0) + (1 if b == "1" else 0)
            new_digit = "1" if total % 2 else "0"
            next_state = "add_loop_carry" if total >= 2 else "add_loop_no_carry"
            add_rule("add_loop_no_carry", (a, "s", b), next_state, (new_digit, "=", "="), ("R", "S", "R"))

    add_rule("add_loop_no_carry", ("_", "s", "_"), "add_return_t1", ("=", "=", "="), ("L", "S", "L"))

    for a in digits:
        for b in digits:
            total = (1 if a == "1" else 0) + (1 if b == "1" else 0) + 1
            new_digit = "1" if total % 2 else "0"
            next_state = "add_loop_carry" if total >= 2 else "add_loop_no_carry"
            add_rule("add_loop_carry", (a, "s", b), next_state, (new_digit, "=", "="), ("R", "S", "R"))

    for symbol in ("0", "1", "_"):
        add_rule("add_return_t1", (symbol, "s", "?"), "add_return_t1", ("=", "=", "="), ("L", "S", "S"))
    add_rule("add_return_t1", ("s", "s", "?"), "add_return_t3", ("=", "=", "="), ("S", "S", "S"))

    for symbol in ("0", "1", "_"):
        add_rule("add_return_t3", ("s", "s", symbol), "add_return_t3", ("=", "=", "="), ("S", "S", "L"))
    add_rule("add_return_t3", ("s", "s", "s"), "dec_start", ("=", "=", "="), ("S", "S", "S"))

    add_rule("dec_start", ("s", "s", "s"), "dec_loop", ("=", "=", "="), ("S", "R", "S"))
    add_rule("dec_loop", ("s", "1", "s"), "dec_trim_move_right", ("=", "0", "="), ("S", "S", "S"))
    add_rule("dec_loop", ("s", "0", "s"), "dec_loop", ("=", "1", "="), ("S", "R", "S"))

    add_rule("dec_trim_move_right", ("s", "0", "s"), "dec_trim_move_right", ("=", "=", "="), ("S", "R", "S"))
    add_rule("dec_trim_move_right", ("s", "1", "s"), "dec_trim_move_right", ("=", "=", "="), ("S", "R", "S"))
    add_rule("dec_trim_move_right", ("s", "_", "s"), "dec_trim_cleanup", ("=", "=", "="), ("S", "L", "S"))

    add_rule("dec_trim_cleanup", ("s", "0", "s"), "dec_trim_cleanup", ("=", "_", "="), ("S", "L", "S"))
    add_rule("dec_trim_cleanup", ("s", "_", "s"), "dec_trim_cleanup", ("=", "=", "="), ("S", "L", "S"))
    add_rule("dec_trim_cleanup", ("s", "1", "s"), "dec_return", ("=", "=", "="), ("S", "S", "S"))
    add_rule("dec_trim_cleanup", ("s", "s", "s"), "dec_return", ("=", "=", "="), ("S", "S", "S"))

    add_rule("dec_return", ("s", "0", "s"), "dec_return", ("=", "=", "="), ("S", "L", "S"))
    add_rule("dec_return", ("s", "1", "s"), "dec_return", ("=", "=", "="), ("S", "L", "S"))
    add_rule("dec_return", ("s", "_", "s"), "dec_return", ("=", "=", "="), ("S", "L", "S"))
    add_rule("dec_return", ("s", "s", "s"), "loop_check", ("=", "=", "="), ("S", "S", "S"))

    add_rule("final_copy_to_t2_prepare", ("s", "s", "s"), "final_copy_to_t2", ("=", "=", "="), ("R", "S", "S"))

    add_rule("final_copy_to_t2", ("0", "s", "s"), "t2_ins_final_start_0", ("_", "=", "="), ("R", "S", "S"))
    add_rule("final_copy_to_t2", ("1", "s", "s"), "t2_ins_final_start_1", ("_", "=", "="), ("R", "S", "S"))
    add_rule("final_copy_to_t2", ("_", "s", "s"), "final_return_t1_after_copy", ("=", "=", "="), ("L", "S", "S"))

    add_rule("final_return_t1_after_copy", ("0", "s", "s"), "final_return_t1_after_copy", ("=", "=", "="), ("L", "S", "S"))
    add_rule("final_return_t1_after_copy", ("1", "s", "s"), "final_return_t1_after_copy", ("=", "=", "="), ("L", "S", "S"))
    add_rule("final_return_t1_after_copy", ("_", "s", "s"), "final_return_t1_after_copy", ("=", "=", "="), ("L", "S", "S"))
    add_rule("final_return_t1_after_copy", ("s", "s", "s"), "final_prepare_write", ("=", "=", "="), ("S", "S", "S"))

    add_rule("final_prepare_write", ("s", "s", "s"), "final_copy_from_t2_start", ("_", "=", "="), ("S", "S", "S"))

    add_rule("final_copy_from_t2_start", ("_", "s", "s"), "final_copy_from_t2", ("=", "=", "="), ("S", "R", "S"))

    add_rule("final_copy_from_t2", ("_", "0", "s"), "final_copy_from_t2", ("0", "=", "="), ("R", "R", "S"))
    add_rule("final_copy_from_t2", ("_", "1", "s"), "final_copy_from_t2", ("1", "=", "="), ("R", "R", "S"))
    add_rule("final_copy_from_t2", ("_", "_", "s"), "final_maybe_zero", ("=", "=", "="), ("L", "L", "S"))

    add_rule("final_maybe_zero", ("?", "?", "s"), "final_trim_t1", ("=", "=", "="), ("S", "S", "S"))

    add_rule("final_trim_t1", ("0", "?", "s"), "final_cleanup_return", ("=", "=", "="), ("S", "S", "S"))
    add_rule("final_trim_t1", ("1", "?", "s"), "final_cleanup_return", ("=", "=", "="), ("S", "S", "S"))
    add_rule("final_trim_t1", ("_", "?", "s"), "final_write_zero", ("0", "=", "="), ("S", "S", "S"))

    add_rule("final_write_zero", ("0", "?", "s"), "final_cleanup_return", ("=", "=", "="), ("S", "S", "S"))

    add_rule("final_cleanup_return", ("?", "0", "s"), "final_cleanup_return", ("=", "=", "="), ("S", "L", "S"))
    add_rule("final_cleanup_return", ("?", "1", "s"), "final_cleanup_return", ("=", "=", "="), ("S", "L", "S"))
    add_rule("final_cleanup_return", ("?", "_", "s"), "final_cleanup_return", ("=", "=", "="), ("S", "L", "S"))
    add_rule("final_cleanup_return", ("?", "s", "s"), "halt", ("=", "=", "="), ("S", "S", "S"))

    return MultiTapeTuringMachine(
        num_tapes=3,
        transitions=transitions,
        initial_state="init",
        halting_state="halt",
        input_vocabulary=["0", "1", "*"],
        blank="_",
    )


__all__ = [
    "binary_addition_machine",
    "binary_multiplication_machine",
]

