"""CoT dimension tests verifying theoretical formulas against actual transformer construction.

Theoretical dimensions from Theorem thm:cotapp (app2_hardmax_constructions.tex):
    L = 5r/2 + 8
    H = 3K
    d_k = 4r - 1
    d_v = max{r, d_Q, d_Gamma}
    d_ff = max{18r + 2, 14Kr + 2r, |Q||Gamma|^K + 1}
    d = 6Kr + 6r + 3*d_Q + (3K+1)*d_Gamma + 10K + 21
"""

from tm.turing import MultiTapeTuringMachine
from turing_to_transformer import turing_machine_to_cot_transformer


def _cartesian_power(items, k):
    if k == 0:
        return [()]
    out = [()]
    for _ in range(k):
        out = [prefix + (x,) for prefix in out for x in items]
    return out


def create_tm(num_states: int, num_symbols: int, num_tapes: int) -> MultiTapeTuringMachine:
    """Create a TM with specified |Q|, |Gamma|, K."""
    states = [f"q{i}" for i in range(num_states)]
    blank = "_"
    input_vocab = [chr(ord("a") + i) for i in range(num_symbols - 1)]
    band_vocab = [blank] + input_vocab

    transitions = {}
    for q in states:
        for syms in _cartesian_power(band_vocab, num_tapes):
            transitions[(q, tuple(syms))] = (states[-1], tuple(syms), tuple("S" for _ in range(num_tapes)))

    return MultiTapeTuringMachine(
        num_tapes=num_tapes,
        transitions=transitions,
        initial_state=states[0],
        halting_state=states[-1],
        input_vocabulary=input_vocab,
        blank=blank,
    )


def theoretical_dimensions(K: int, d_Q: int, d_Gamma: int, r: int, num_states: int, num_symbols: int) -> dict:
    """Compute theoretical CoT dimensions from Theorem thm:cotapp."""
    # d = 6Kr + 6r + 3*d_Q + (3K+1)*d_Gamma + 10K + 21
    d = 6 * K * r + 6 * r + 3 * d_Q + (3 * K + 1) * d_Gamma + 10 * K + 21

    # d_ff = max{18r + 2, 14Kr + 2r, |Q||Gamma|^K + 1}
    term1 = 18 * r + 2
    term2 = 14 * K * r + 2 * r
    term3 = num_states * (num_symbols ** K) + 1
    d_ff = max(term1, term2, term3)

    d_k = 4 * r - 1
    d_v = max(r, d_Q, d_Gamma)
    H = 3 * K

    # L = 5r/2 + 8, computed via layer indices
    L_1 = r // 2 + 1
    L_2 = L_1 + r + 2
    L_3 = L_2 + r + 1
    L = L_3 + 4

    return {
        "d": d,
        "d_ff": d_ff,
        "d_k": d_k,
        "d_v": d_v,
        "H": H,
        "L": L,
        "term1": term1,
        "term2": term2,
        "term3": term3,
    }


def get_actual_dimensions(t) -> dict:
    """Extract actual dimensions from constructed transformer."""
    d = t.embedding_dim
    max_dff = max(mlp.num_neurons for mlp in t._mlps_by_layer.values())
    max_H = max(len(heads) for heads in t._heads_by_layer.values())
    max_dk = max(len(h[0]) for heads in t._heads_by_layer.values() for h in heads)
    max_dv = max(len(h[2]) for heads in t._heads_by_layer.values() for h in heads)
    L = int(max(t._mlps_by_layer.keys()))
    return {"d": d, "d_ff": max_dff, "d_k": max_dk, "d_v": max_dv, "H": max_H, "L": L}


# Test cases designed to hit each d_ff term, with varied |Q| for different d_Q values.
TEST_CASES = [
    # (num_states, num_symbols, K, r)
    # term1 dominant (18r+2): K=1, small term3, varied |Q| for d_Q coverage
    (2, 2, 1, 4),    # d_Q=1
    (4, 2, 1, 6),    # d_Q=2
    (8, 2, 1, 8),    # d_Q=3
    (12, 2, 1, 10),  # d_Q=4
    (3, 3, 1, 4),    # d_Q=2
    # term2 dominant (14Kr+2r): K>=2, small term3, varied |Q|
    (2, 2, 2, 4),    # d_Q=1
    (4, 2, 2, 6),    # d_Q=2
    (8, 2, 2, 8),    # d_Q=3
    (2, 2, 3, 4),
    (4, 2, 3, 6),    # d_Q=2, K=3
    (2, 2, 4, 4),
    # term3 dominant (transitions): larger |Q| or |Gamma|
    (10, 5, 2, 4),   # 10*25=250
    (4, 8, 2, 4),    # 4*64=256
    (8, 8, 1, 4),    # d_Q=3
    (20, 2, 1, 4),   # d_Q=5
    (2, 20, 1, 4),   # d_Gamma=5
    (15, 3, 2, 4),   # 15*9=135
    (3, 8, 2, 4),    # 3*64=192
    # Balanced / edge cases with varied |Q|
    (5, 4, 2, 4),    # d_Q=3
    (7, 3, 2, 4),    # d_Q=3
    (6, 3, 2, 4),
]


def test_cot_dimensions():
    """Test CoT dimensions against theoretical formulas."""
    for num_states, num_symbols, K, r in TEST_CASES:
        d_Q = max(1, (num_states - 1).bit_length())
        d_Gamma = max(1, (num_symbols - 1).bit_length())

        tm = create_tm(num_states, num_symbols, K)
        t = turing_machine_to_cot_transformer(tm, r=r)

        actual = get_actual_dimensions(t)
        theory = theoretical_dimensions(K, d_Q, d_Gamma, r, num_states, num_symbols)

        for name in ["d", "L", "H", "d_k", "d_v", "d_ff"]:
            assert actual[name] == theory[name], (
                f"|Q|={num_states}, |Gamma|={num_symbols}, K={K}, r={r}: "
                f"{name} mismatch: actual={actual[name]}, theory={theory[name]}"
            )


def main():
    """Run tests with verbose output."""
    print("CoT Dimension Tests")
    print("=" * 100)

    all_pass = True
    for num_states, num_symbols, K, r in TEST_CASES:
        d_Q = max(1, (num_states - 1).bit_length())
        d_Gamma = max(1, (num_symbols - 1).bit_length())

        tm = create_tm(num_states, num_symbols, K)
        t = turing_machine_to_cot_transformer(tm, r=r)

        actual = get_actual_dimensions(t)
        theory = theoretical_dimensions(K, d_Q, d_Gamma, r, num_states, num_symbols)

        checks = [
            ("d", actual["d"], theory["d"]),
            ("L", actual["L"], theory["L"]),
            ("H", actual["H"], theory["H"]),
            ("d_k", actual["d_k"], theory["d_k"]),
            ("d_v", actual["d_v"], theory["d_v"]),
            ("d_ff", actual["d_ff"], theory["d_ff"]),
        ]

        failures = [(n, a, th) for n, a, th in checks if a != th]
        status = "PASS" if not failures else "FAIL"
        if failures:
            all_pass = False

        t1, t2, t3 = theory["term1"], theory["term2"], theory["term3"]
        max_term = max(t1, t2, t3)
        dominant = []
        if t1 == max_term:
            dominant.append("t1")
        if t2 == max_term:
            dominant.append("t2")
        if t3 == max_term:
            dominant.append("t3")

        print(
            f"|Q|={num_states:2}, |G|={num_symbols:2}, K={K}, r={r:2} | {status} | "
            f"d_ff={actual['d_ff']:4} (t1={t1:4}, t2={t2:4}, t3={t3:4}) [{'/'.join(dominant)}]"
        )

        if failures:
            for name, act, thy in failures:
                print(f"  MISMATCH {name}: actual={act}, theory={thy}")

    print("=" * 100)
    print("All tests PASSED!" if all_pass else "Some tests FAILED!")
    return 0 if all_pass else 1


if __name__ == "__main__":
    main()
