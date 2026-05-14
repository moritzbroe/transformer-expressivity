import torch

from transformer import MOVE_ENC_2D, Transformer, int_to_signed_binary_lsb


def test_positional_encoding_uses_first_register():
    t = Transformer(
        vocab=["a", "b"],
        registers=[("pos", 3), ("x", 2)],
        flags=["f"],
    )
    x = t.forward(["a", "b", "a", "a"])
    pos_sl = t.register_slices["pos"]
    x_sl = t.register_slices["x"]

    expected_pos = torch.tensor([int_to_signed_binary_lsb(i, 3) for i in range(4)], dtype=torch.float32)
    torch.testing.assert_close(x[:, pos_sl], expected_pos)
    torch.testing.assert_close(x[:, x_sl], torch.zeros((4, 2)))


def test_set_register_and_flag_embeddings():
    t = Transformer(
        vocab=["a", "b"],
        registers=[("pos", 2), ("x", 2)],
        flags=["f"],
    )
    t.set_register_embeddings("x", {"a": [1.0, -1.0], "b": [-1.0, 1.0]})
    t.set_flag_embeddings("f", {"a": 1.0, "b": 0.0})

    x = t.forward(["a", "b"])
    torch.testing.assert_close(x[:, t.register_slices["x"]], torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))
    torch.testing.assert_close(x[:, t.flag_indices["f"]], torch.tensor([1.0, 0.0]))


def test_attention_head_copies_value_to_out_register():
    t = Transformer(
        vocab=["t0", "t1", "t2"],
        registers=[("pos", 2), ("k", 2), ("v", 2), ("out", 2)],
        flags=[],
    )
    t.set_register_embeddings("k", {"t0": [1.0, -1.0], "t1": [-1.0, 1.0], "t2": [1.0, 1.0]})
    t.set_register_embeddings("v", {"t0": [1.0, 0.0], "t1": [0.0, 1.0], "t2": [1.0, 1.0]})

    t.add_head(layer=1, q=["k", "k"], k=["k", "k"], v=["v"], out="out")
    x = t.forward(["t0", "t1", "t2"])
    torch.testing.assert_close(x[:, t.register_slices["out"]], x[:, t.register_slices["v"]])


def test_mlp_copy_and_zero_with_conditionals():
    t = Transformer(
        vocab=["on", "off"],
        registers=[("pos", 1), ("a", 2), ("b", 2)],
        flags=["flag"],
    )
    t.set_register_embeddings("a", {"on": [1.0, -1.0], "off": [-1.0, 1.0]})
    t.set_flag_embeddings("flag", {"on": 1.0, "off": 0.0})

    t.mlp_copy(layer=1, src="a", dst="b", when=["flag"])
    x = t.forward(["on", "off"])
    torch.testing.assert_close(x[0, t.register_slices["b"]], x[0, t.register_slices["a"]])
    torch.testing.assert_close(x[1, t.register_slices["b"]], torch.zeros(2))

    t2 = Transformer(
        vocab=["tok"],
        registers=[("pos", 1), ("b", 2)],
        flags=[],
    )
    t2.set_register_embeddings("b", {"tok": [1.0, -1.0]})
    t2.mlp_zero(layer=1, target="b")
    x2 = t2.forward(["tok"])
    torch.testing.assert_close(x2[0, t2.register_slices["b"]], torch.zeros(2))


def test_mlp_subtract_power_of_two_and_full_subtraction():
    t = Transformer(
        vocab=["tok4", "tok0"],
        registers=[("pos", 1), ("n", 3), ("out", 3)],
        flags=[],
    )
    t.set_register_embeddings("n", {"tok4": [-1.0, -1.0, 1.0], "tok0": [-1.0, -1.0, -1.0]})  # 4 and 0
    t.mlp_subtract_power_of_two(layer=1, inp="n", out="out", k=0)
    x = t.forward(["tok4", "tok0"])
    torch.testing.assert_close(x[0, t.register_slices["out"]], torch.tensor([1.0, 1.0, -1.0]))  # 4-1 = 3
    torch.testing.assert_close(x[1, t.register_slices["out"]], torch.tensor([-1.0, -1.0, -1.0]))  # saturates at 0

    t2 = Transformer(
        vocab=["tok"],
        registers=[("pos", 1), ("a", 3), ("b", 3)],
        flags=["F"],
    )
    t2.set_register_embeddings("a", {"tok": [1.0, 1.0, -1.0]})  # 3
    t2.set_register_embeddings("b", {"tok": [-1.0, -1.0, 1.0]})  # 4
    t2.set_flag_embeddings("F", {"tok": 1.0})

    t2.mlp_full_subtraction(layer=1, subtrahend="a", minuend_inplace="b", when=["F"])
    x2 = t2.forward(["tok"])
    torch.testing.assert_close(x2[0, t2.register_slices["b"]], torch.tensor([1.0, -1.0, -1.0]))  # 4-3 = 1


def test_mlp_add_move_r_and_l():
    t = Transformer(
        vocab=["tok_r", "tok_l"],
        registers=[("pos", 1), ("p", 2), ("mv", 2), ("out", 2)],
        flags=[],
    )
    t.set_register_embeddings("p", {"tok_r": [1.0, -1.0], "tok_l": [-1.0, -1.0]})  # 1 and 0
    t.set_register_embeddings("mv", {"tok_r": list(MOVE_ENC_2D["R"]), "tok_l": list(MOVE_ENC_2D["L"])})
    t.mlp_add_head_movement(layer=1, inp="p", move="mv", out="out")

    x = t.forward(["tok_r", "tok_l"])
    torch.testing.assert_close(x[0, t.register_slices["out"]], torch.tensor([-1.0, 1.0]))  # 1 + 1 = 2
    torch.testing.assert_close(x[1, t.register_slices["out"]], torch.tensor([-1.0, -1.0]))  # 0 - 1 saturates at 0


if __name__ == "__main__":
    print("Transformer Tests")
    print("=" * 80)
    test_positional_encoding_uses_first_register()
    print("test_positional_encoding_uses_first_register passed")
    test_set_register_and_flag_embeddings()
    print("test_set_register_and_flag_embeddings passed")
    test_attention_head_copies_value_to_out_register()
    print("test_attention_head_copies_value_to_out_register passed")
    test_mlp_copy_and_zero_with_conditionals()
    print("test_mlp_copy_and_zero_with_conditionals passed")
    test_mlp_subtract_power_of_two_and_full_subtraction()
    print("test_mlp_subtract_power_of_two_and_full_subtraction passed")
    test_mlp_add_move_r_and_l()
    print("test_mlp_add_move_r_and_l passed")
    print("=" * 80)
    print("All tests passed!")
