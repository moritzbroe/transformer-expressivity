from tm.turing import MultiTapeTuringMachine, Tape, Configuration
from typing import Tuple
from tm.examples import binary_addition_machine
from tm.examples import binary_multiplication_machine
import random

def run_machine_cases(tm: MultiTapeTuringMachine, cases: Tuple[str, str]):
    input_vocab = set(tm.input_vocabulary)
    states = tm.states
    assert tm.initial_state in states, "initial state missing from machine states"
    assert tm.halting_state in states, "halting state missing from machine states"
    for input_str, expected_output in cases:
        assert set(input_str).issubset(input_vocab), f"Input {input_str} uses symbols outside the machine input vocabulary"
        try:
            tm_output = tm.run(input_str, max_steps=10**6)
        except Exception as e:
            raise AssertionError(f"Input: {input_str}, Expected: {expected_output}, Got RuntimeError: {str(e)}")

        if isinstance(tm_output, tuple):
            tm_value, _ = tm_output
        else:
            tm_value = tm_output

        assert tm_value == expected_output, f"Input: {input_str}, Expected: {expected_output}, Got: {tm_output}"
    print("All test cases passed.")


def test_binary_addition_machine():
    tm = binary_addition_machine()
    cases = [
        ("110+101", "1011"),  # 6 + 5 = 11
        ("0+0", "0"),         # 0 + 0 = 0
        ("1+1", "10"),        # 1 + 1 = 2
        ("111+1", "1000"),    # 7 + 1 = 8
        ("1010+110", "10000"),# 10 + 6 = 16
    ]
    for i in range(100):
        len_a = random.randint(1, 10)
        len_b = random.randint(1, 10)
        a = random.randint(0, 2**len_a - 1)
        b = random.randint(0, 2**len_b - 1)
        input_str = f"{bin(a)[2:]}+{bin(b)[2:]}"
        expected_output = bin(a + b)[2:]
        cases.append((input_str, expected_output))
    run_machine_cases(tm, cases)


def test_binary_multiplication_machine():
    tm = binary_multiplication_machine()
    cases = [
        ("0*0", "0"),           # 0 * 0 = 0
        ("1*0", "0"),           # 1 * 0 = 0
        ("1*1", "1"),           # 1 * 1 = 1
        ("10*10", "100"),       # 2 * 2 = 4
        ("101*11", "1111"),     # 5 * 3 = 15
        ("111*101", "100011"),  # 7 * 5 = 35
    ]
    for i in range(100):
        len_a = random.randint(1, 10)
        len_b = random.randint(1, 10)
        a = random.randint(0, 2**len_a - 1)
        b = random.randint(0, 2**len_b - 1)
        input_str = f"{bin(a)[2:]}*{bin(b)[2:]}"
        expected_output = bin(a * b)[2:]
        cases.append((input_str, expected_output))
    run_machine_cases(tm, cases)


if __name__ == "__main__":
    print("Turing Machine Tests")
    print("=" * 80)
    print("Testing binary addition machine...")
    test_binary_addition_machine()
    print("Testing binary multiplication machine...")
    test_binary_multiplication_machine()
    print("=" * 80)
