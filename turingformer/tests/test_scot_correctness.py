"""SCoT correctness tests: verify transformer produces correct token sequences.

Run with: python -m tests.test_scot_correctness
"""

import random

from tm.random_tm import random_tm_and_input
from tm.turing import MultiTapeTuringMachine
from turing_to_transformer import EINP, ESUMM, OUTP, EOUTP, scot_token_segments, turing_machine_to_scot_transformer


def test_empty_output():
    """Test SCoT with TMs that produce empty output."""
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
        assert output == ""
        assert steps >= 1

        segments = scot_token_segments(tm, r=4, input_word=input_word, max_steps=100)
        last_seg = segments[-1]
        assert last_seg[-2] == OUTP
        assert last_seg[-1] == EOUTP

        t = turing_machine_to_scot_transformer(tm, r=4)
        for seg in segments:
            preds = t.predict_all(seg[:-1])
            boundary = next(i for i, tok in enumerate(seg) if tok in {EINP, ESUMM})
            for i in range(boundary, len(seg) - 1):
                assert preds[i] == seg[i + 1], f"input={input_word!r}, i={i}, token={seg[i]}, pred={preds[i]}, expected={seg[i + 1]}"

    print("test_empty_output passed")


def test_empty_input():
    """Test SCoT with empty input."""
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
    assert output == "a"

    segments = scot_token_segments(tm, r=4, input_word=input_word, max_steps=100)
    t = turing_machine_to_scot_transformer(tm, r=4)

    for seg in segments:
        preds = t.predict_all(seg[:-1])
        boundary = next(i for i, tok in enumerate(seg) if tok in {EINP, ESUMM})
        for i in range(boundary, len(seg) - 1):
            assert preds[i] == seg[i + 1], f"i={i}, token={seg[i]}, pred={preds[i]}, expected={seg[i + 1]}"

    print("test_empty_input passed")


def test_empty_input_and_output():
    """Test SCoT with both empty input and empty output."""
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

    segments = scot_token_segments(tm, r=4, input_word=input_word, max_steps=100)
    t = turing_machine_to_scot_transformer(tm, r=4)

    for seg in segments:
        preds = t.predict_all(seg[:-1])
        boundary = next(i for i, tok in enumerate(seg) if tok in {EINP, ESUMM})
        for i in range(boundary, len(seg) - 1):
            assert preds[i] == seg[i + 1], f"i={i}, token={seg[i]}, pred={preds[i]}, expected={seg[i + 1]}"

    print("test_empty_input_and_output passed")


def test_random_tms(num_iterations: int = 5000):
    """Test SCoT on random Turing machines."""
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

        r_min = None
        last_exc = None
        for r in range(4, 62, 2):
            try:
                segments = scot_token_segments(tm, r=r, input_word=input_word, max_steps=10000)
            except Exception as exc:
                last_exc = exc
                continue
            if max(len(seg) for seg in segments) <= (1 << r):
                r_min = r
                break

        if r_min is None:
            num_skipped += 1
            if last_exc is None:
                print(f"{iteration} skipped: no valid r found", flush=True)
            else:
                print(f"{iteration} skipped: no valid r found (last error: {type(last_exc).__name__}: {last_exc})", flush=True)
            continue

        r = r_min
        segments = scot_token_segments(tm, r=r, input_word=input_word, max_steps=10000)
        t = turing_machine_to_scot_transformer(tm, r=r)

        for seg in segments:
            preds = t.predict_all(seg[:-1])
            boundary = next(i for i, tok in enumerate(seg) if tok in {EINP, ESUMM})
            for i in range(boundary, len(seg) - 1):
                assert preds[i] == seg[i + 1], (iteration, i, seg[i], preds[i], seg[i + 1])

        num_correct += 1
        segment_lengths = [len(seg) for seg in segments]
        print(
            f"{iteration} correct: {len(tm.states)} states, {len(tm.band_vocabulary)} band vocab, {tm.num_tapes} tapes, "
            f"input {input_word}, output {output_word}, r={r}, segments={len(segments)}, segment_lengths={segment_lengths}",
            flush=True,
        )

    print(f"\nSummary: {num_correct}/{num_iterations} correct, {num_skipped} skipped")
    return num_correct > 0


if __name__ == "__main__":
    print("SCoT Correctness Tests")
    print("=" * 80)

    test_empty_output()
    test_empty_input()
    test_empty_input_and_output()

    print("\nRunning random TM tests...")
    test_random_tms(num_iterations=5000)

    print("=" * 80)
