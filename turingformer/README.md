# Turing Machine to Transformer Construction

This folder contains a PyTorch implementation that compiles multi-tape Turing machines into hardmax-attention Transformer decoders and verifies that the resulting model generates the intended CoT/SCoT token sequences.

All commands below should be run from this directory (`turingformer/`).

## Setup

Create a conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate turingformer
```

Alternatively, install manually: Python 3.11+, PyTorch (CPU is sufficient).

## Running the Tests

### CoT and SCoT Correctness

These tests verify that the constructed transformer produces the correct token sequences for various Turing machines:

```bash
python -m tests.test_cot_correctness
python -m tests.test_scot_correctness
```

Each runs edge-case tests (empty input/output) followed by 5000 random Turing machines. Many random TMs don't halt or halt with output containing blanks; these are skipped (but printed) as they are not covered by the construction.

### CoT and SCoT Dimensions

These tests verify that the constructed transformer dimensions match the theoretical formulas from the paper:

```bash
python -m tests.test_cot_dimensions
python -m tests.test_scot_dimensions
```

### Turing Machine Tests

Tests for the example Turing machines (binary addition/multiplication):

```bash
python -m tests.test_tm
```
