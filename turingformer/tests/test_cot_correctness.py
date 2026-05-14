"""CoT correctness tests: verify transformer produces correct token sequences.

Run with: python -m tests.test_cot_correctness
"""

import random

from tm.random_tm import random_tm_and_input
from tm.turing import MultiTapeTuringMachine
from turing_to_transformer import EINP, OUTP, EOUTP, cot_token_sequence, turing_machine_to_cot_transformer


def test_empty_output():
    """Test CoT with TMs that produce empty output (tape is all blanks)."""
    tm = MultiTapeTuringMachine(
        num_tapes=1,
        transitions={
            ("q0", ("a",)): ("q0", ("_",), ("R",)),
            ("q0", ("b",)): ("q0", ("_",), ("R",)),
            ("q0", ("_",)): ("halt", ("_",), ("S",)),
        },
        initial_state="q0",
        halting_state="halt",
        input_vocabulary=["a", "b"],
        blank="_",
    )

    for input_word in ["a", "b", "aa", ""]:
        output, steps = tm.run(input_word, max_steps=100)
        assert output == "", f"Expected empty output, got {output!r}"
        assert steps >= 1

        toks = cot_token_sequence(tm, r=4, input_word=input_word, max_steps=100)
        assert toks[-2] == OUTP
        assert toks[-1] == EOUTP

        t = turing_machine_to_cot_transformer(tm, r=4)
        preds = t.predict_all(toks[:-1])

        einp_index = 1 + len(input_word)
        assert toks[einp_index] == EINP
        for i in range(einp_index, len(toks) - 1):
            assert preds[i] == toks[i + 1], f"input={input_word!r}, i={i}, token={toks[i]}, pred={preds[i]}, expected={toks[i + 1]}"

    print("test_empty_output passed")


def test_empty_input():
    """Test CoT with empty input."""
    tm = MultiTapeTuringMachine(
        num_tapes=1,
        transitions={
            ("q0", ("a",)): ("halt", ("a",), ("S",)),
            ("q0", ("b",)): ("halt", ("b",), ("S",)),
            ("q0", ("_",)): ("halt", ("a",), ("S",)),
        },
        initial_state="q0",
        halting_state="halt",
        input_vocabulary=["a", "b"],
        blank="_",
    )

    input_word = ""
    output, steps = tm.run(input_word, max_steps=100)
    assert output == "a", f"Expected 'a', got {output!r}"
    assert steps == 1

    toks = cot_token_sequence(tm, r=4, input_word=input_word, max_steps=100)
    assert toks[0] == "inp"
    assert toks[1] == EINP
    assert toks[-3] == OUTP
    assert toks[-2] == "a"
    assert toks[-1] == EOUTP

    t = turing_machine_to_cot_transformer(tm, r=4)
    preds = t.predict_all(toks[:-1])

    einp_index = 1
    assert toks[einp_index] == EINP
    for i in range(einp_index, len(toks) - 1):
        assert preds[i] == toks[i + 1], f"i={i}, token={toks[i]}, pred={preds[i]}, expected={toks[i + 1]}"

    print("test_empty_input passed")


def test_empty_input_and_output():
    """Test CoT with both empty input and empty output."""
    tm = MultiTapeTuringMachine(
        num_tapes=1,
        transitions={
            ("q0", ("a",)): ("halt", ("_",), ("S",)),
            ("q0", ("b",)): ("halt", ("_",), ("S",)),
            ("q0", ("_",)): ("halt", ("_",), ("S",)),
        },
        initial_state="q0",
        halting_state="halt",
        input_vocabulary=["a", "b"],
        blank="_",
    )

    input_word = ""
    output, steps = tm.run(input_word, max_steps=100)
    assert output == ""
    assert steps == 1

    toks = cot_token_sequence(tm, r=4, input_word=input_word, max_steps=100)
    assert toks[0] == "inp"
    assert toks[1] == EINP
    assert toks[-2] == OUTP
    assert toks[-1] == EOUTP

    t = turing_machine_to_cot_transformer(tm, r=4)
    preds = t.predict_all(toks[:-1])

    einp_index = 1
    for i in range(einp_index, len(toks) - 1):
        assert preds[i] == toks[i + 1], f"i={i}, token={toks[i]}, pred={preds[i]}, expected={toks[i + 1]}"

    print("test_empty_input_and_output passed")


def test_random_tms(num_iterations: int = 5000):
    """Test CoT on random Turing machines."""
    rng = random.Random(0)

    num_skipped = 0
    num_correct = 0

    for iteration in range(1, num_iterations + 1):
        tm, input_word = random_tm_and_input(
            rng=rng,
            num_tapes_choices=(1, 3),
            num_states_range=(2, 10),
            sigma_choices=("a", "b", "c", "d", "e"),
            sigma_size_range=(1, 5),
            input_len_range=(1, 12),
            blank="_",
            halt_transition_prob=0.05,
        )

        try:
            output_word, steps = tm.run(input_word, max_steps=10000)
        except Exception as exc:
            num_skipped += 1
            if isinstance(exc, RuntimeError) and str(exc) == "too many steps":
                print(f"{iteration} skipped: tm didn't halt within 10000 steps", flush=True)
            else:
                print(f"{iteration} skipped: tm.run raised {type(exc).__name__}: {exc}", flush=True)
            continue

        if steps <= 0:
            num_skipped += 1
            print(f"{iteration} skipped: non-positive steps", flush=True)
            continue
        if any(ch not in tm.input_vocabulary for ch in output_word):
            num_skipped += 1
            print(f"{iteration} skipped: output contains symbols not in input vocabulary", flush=True)
            continue

        def cot_len(r: int) -> int:
            j_max = ((steps + r - 1) // r) - 1
            return (1 + len(input_word) + 1) + j_max * (2 * r + 2) + (steps - r * j_max) + (1 + len(output_word) + 1)

        r_min = None
        for r in range(2, 62, 2):
            if cot_len(r) <= (1 << r):
                r_min = r
                break

        if r_min is None:
            num_skipped += 1
            print(f"{iteration} skipped: no valid r found", flush=True)
            continue

        r = r_min + rng.choice([0, 2, 4, 6, 8])

        try:
            toks = cot_token_sequence(tm, r=r, input_word=input_word, max_steps=10000)
        except Exception as exc:
            num_skipped += 1
            print(f"{iteration} skipped: cot_token_sequence raised {type(exc).__name__}: {exc}", flush=True)
            continue

        t = turing_machine_to_cot_transformer(tm, r=r)
        preds = t.predict_all(toks[:-1])

        einp_index = 1 + len(input_word)
        assert toks[einp_index] == EINP
        for i in range(einp_index, len(toks) - 1):
            assert preds[i] == toks[i + 1], (iteration, i, toks[i], preds[i], toks[i + 1])

        num_correct += 1
        print(
            f"{iteration} correct: {len(tm.states)} states, {len(tm.band_vocabulary)} band vocab, {tm.num_tapes} tapes, "
            f"input {input_word}, output {output_word}, cot length {len(toks)}, r={r}",
            flush=True,
        )

    print(f"\nSummary: {num_correct}/{num_iterations} correct, {num_skipped} skipped")
    return num_correct > 0


if __name__ == "__main__":
    print("CoT Correctness Tests")
    print("=" * 80)

    test_empty_output()
    test_empty_input()
    test_empty_input_and_output()

    print("\nRunning random TM tests...")
    test_random_tms(num_iterations=5000)

    print("=" * 80)
