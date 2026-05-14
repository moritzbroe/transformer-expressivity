# Sudoku reasoning (CoT / SCoT)

Generates Sudoku solver traces, trains small decoder-only transformers (GPT-NeoX style) on them, and evaluates by autoregressive generation. All commands are run from the `sudoku/` directory and require one CUDA GPU for training and evaluation.

## Setup

```bash
conda env create -f environment.yml
conda activate sudoku-reasoning
```

## Data

Download the [sudoku-extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme) puzzles, precompute test CoT lengths, and generate training data for both modes:

```bash
mkdir -p data
curl -L -o data/train.csv 'https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/train.csv?download=true'
curl -L -o data/test.csv 'https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/test.csv?download=true'
python -m sudoku_reasoning.clean_sudoku_extreme

python -m sudoku_reasoning.precompute_cot_lengths \
  --input data/test.txt --output data/test_cot_lengths.json \
  --max-solver-steps 10000000 --workers 64

python -m sudoku_reasoning.generate_train_data --mode cot \
  --input data/train.txt --output data/train_data/cot_16k \
  --max-cot-length 16384 --workers 64
python -m sudoku_reasoning.generate_train_data --mode scot \
  --input data/train.txt --output data/train_data/scot_16k \
  --segment-trace-tokens 512 --max-cot-length 16384 --workers 64
```

## Train

Baseline CoT run (`L=6, d=512, H=8`, trained on puzzles with CoT length ≤ 2^10):

```bash
python -m sudoku_reasoning.train \
  --train-data data/train_data/cot_16k \
  --output-dir checkpoints/cot_l6_1k \
  --tokens 20000000000 --tokens-per-batch 100000 \
  --learning-rate 5e-4 --floor-factor 0.02 --num-workers 16 \
  --num-layers 6 --num-heads 8 --hidden-size 512 \
  --max-cot-tokens 1024
```

For larger `--max-cot-tokens`, use `--tokens-per-batch 25000 --grad-accum 4`. Same 100k tokens per optimizer step, split into 4 length-bucketed micro-batches — reduces padding without changing behavior.

For SCoT, swap `--train-data data/train_data/scot_16k`; segments are bounded by `--segment-trace-tokens`, so `--grad-accum` is not needed.

To reproduce specific paper experiments, change these flags from the baseline:

| Experiment | Flag change |
| --- | --- |
| Training cap sweep (§5.2)    | `--max-cot-tokens N` for `N ∈ {1024, 2048, 4096, 8192, 16384}` |
| SCoT depth sweep (§5.2)      | `--num-layers L` for `L ∈ {4,5,6,7,8}` (SCoT mode) |
| Increased model depth (§5.3) | `--num-layers 8` |
| fp32 activations (§5.3)      | add `--fp32` (forces full fp32, disables TF32) |
| Larger SCoT model (App E.5)  | `--hidden-size 768 --num-layers 12 --num-heads 12` (SCoT mode) |
| Longer training (App E.4)    | `--tokens 50000000000` |
| Hardest-100 eval (App E.5)   | run eval with `--targets 10000000 --per-target 100` (picks the 100 test puzzles closest to that target — effectively the hardest 100) |

## Evaluate

Evaluation expects `data/test_cot_lengths.json` (generated above). CoT evaluation uses vLLM; SCoT generates segment-wise.

Random subsample accuracy:

```bash
python -m sudoku_reasoning.eval_cot checkpoints/cot_l6_1k \
  --data-path data/test.txt --count 10000 --max-new-tokens 32512
python -m sudoku_reasoning.eval_scot checkpoints/scot_l6_1k \
  --data-path data/test.txt --count 10000 --batch-size 128 \
  --max-new-tokens 10000000 --max-segment-length 2048
```

Target-based evaluation for accuracy-vs-CoT-length plots:

```bash
python -m sudoku_reasoning.eval_cot checkpoints/cot_l6_1k \
  --data-path data/test.txt --per-target 100 \
  --targets-logspace 512 23170 23 --max-new-tokens 32512 > cot_targets.log
python -m sudoku_reasoning.eval_scot checkpoints/scot_l6_1k \
  --data-path data/test.txt --per-target 100 --batch-size 128 \
  --targets-logspace 512 23170 23 --max-new-tokens 65536 \
  --max-segment-length 2048 > scot_targets.log

python -m sudoku_reasoning.plot_eval_results cot_targets.log scot_targets.log \
  --labels CoT SCoT --output eval_targets.png --logx
```
