import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, Trainer, TrainingArguments
from transformers.trainer_pt_utils import get_length_grouped_indices

from sudoku_reasoning.precision_utils import add_fp32_arg, get_precision
from sudoku_reasoning.tokenizer import SudokuTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Sudoku decoder model")
    add_fp32_arg(parser)
    parser.add_argument(
        "--train-data",
        default="data/train_data",
        help="Directory created by generate_train_data.py (Arrow dataset + metadata)",
    )
    parser.add_argument("--output-dir", default="checkpoints/trash")
    parser.add_argument(
        "--tokens",
        type=float,
        required=True,
        help="Target number of non-pad tokens to train on (uses unpadded sequence lengths).",
    )
    parser.add_argument(
        "--tokens-per-batch",
        type=int,
        required=True,
        help="Target number of non-pad tokens per batch; batch size is chosen from the dataset average length.",
    )
    parser.add_argument(
        "--max-cot-tokens",
        type=int,
        default=None,
        help="Train only on samples with cot_length <= this value (defaults to dataset max_cot_length).",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--floor-factor",
        type=float,
        default=0.0,
        help="Cosine LR floor ratio (0.0 keeps standard cosine schedule).",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument(
        "--num-heads",
        type=int,
        default=12,
        help="Must divide hidden-size evenly",
    )
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--init-from", default=None, help="Load model weights from checkpoint (fresh training state)")
    return parser.parse_args()


def build_model(args, tokenizer: SudokuTokenizer, context_length: int):
    if args.hidden_size % args.num_heads != 0:
        raise ValueError("hidden-size must be divisible by num-heads for attention")
    config = GPTNeoXConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=4 * args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        max_position_embeddings=context_length,
        rope_theta=10000.0,
        rotary_pct=1.0,
        bos_token_id=tokenizer.input_id,
        eos_token_id=tokenizer.output_end_id,
        pad_token_id=tokenizer.pad_id,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    return GPTNeoXForCausalLM(config)


def load_training_data(path: str):
    data_dir = Path(path)
    if not data_dir.exists():
        raise FileNotFoundError(f"{path} does not exist")

    dataset = load_from_disk(str(data_dir))
    metadata = {}
    meta_path = data_dir / "metadata.json"
    if meta_path.exists():
        with meta_path.open() as handle:
            metadata.update(json.load(handle))

    return dataset, metadata


def main():
    args = parse_args()
    if torch.cuda.device_count() != 1:
        raise SystemExit(f"Expected exactly 1 GPU, found {torch.cuda.device_count()}")
    grad_accum_was_set = args.grad_accum is not None
    group_by_length = grad_accum_was_set
    if args.grad_accum is None:
        args.grad_accum = 1
    if args.grad_accum <= 0:
        raise SystemExit("--grad-accum must be > 0")
    if args.tokens_per_batch <= 0:
        raise SystemExit("--tokens-per-batch must be > 0")
    os.makedirs(args.output_dir, exist_ok=True)

    dataset, metadata = load_training_data(args.train_data)
    dataset_max_cot_length = metadata.get("max_cot_length")
    if dataset_max_cot_length is None:
        raise SystemExit("Training data metadata missing max_cot_length; regenerate with updated generator.")
    dataset_max_cot_length = int(dataset_max_cot_length)
    train_max_cot_tokens = dataset_max_cot_length if args.max_cot_tokens is None else int(args.max_cot_tokens)
    if train_max_cot_tokens <= 0:
        raise SystemExit("--max-cot-tokens must be > 0")
    if train_max_cot_tokens > dataset_max_cot_length:
        raise SystemExit(
            f"--max-cot-tokens ({train_max_cot_tokens}) exceeds dataset max_cot_length ({dataset_max_cot_length})"
        )
    lengths = list(dataset["length"])
    cot_lengths = list(dataset["cot_length"])
    if len(lengths) != len(cot_lengths):
        raise SystemExit("Training data columns length and cot_length have mismatched sizes; regenerate dataset.")
    keep_indices = [idx for idx, cot_len in enumerate(cot_lengths) if int(cot_len) <= train_max_cot_tokens]
    if not keep_indices:
        raise SystemExit(f"No samples left after filtering to cot_length <= {train_max_cot_tokens}")
    total_tokens = float(sum(int(lengths[idx]) for idx in keep_indices))
    avg_tokens_per_sample = total_tokens / len(keep_indices)
    per_device_batch_size = max(1, int(args.tokens_per_batch / avg_tokens_per_sample))
    avg_tokens_per_batch = per_device_batch_size * avg_tokens_per_sample
    avg_tokens_per_update = avg_tokens_per_batch * args.grad_accum
    msg = (
        f"Segments={len(keep_indices)} total_tokens={int(total_tokens)} "
        f"avg_tokens_per_sample={avg_tokens_per_sample:.1f} "
        f"tokens_per_batch_target={args.tokens_per_batch} "
        f"batch_size={per_device_batch_size} "
    )
    if grad_accum_was_set:
        msg += f"grad_accum={args.grad_accum} avg_tokens_per_batch={avg_tokens_per_batch:.1f} "
    msg += f"avg_tokens_per_update={avg_tokens_per_update:.1f}"
    print(msg, flush=True)
    dataset = dataset.select(keep_indices)
    tokenizer = SudokuTokenizer()
    pad_id = metadata.get("pad_id", tokenizer.pad_id)
    context_length = 32768

    if args.init_from:
        model = GPTNeoXForCausalLM.from_pretrained(args.init_from)
        print(f"Loaded model weights from {args.init_from}", flush=True)
    else:
        model = build_model(args, tokenizer, context_length)
    print(f"Attention implementation: {getattr(model.config, '_attn_implementation', None)}", flush=True)
    precision = get_precision(force_fp32=args.fp32)

    def collate_fn(batch):
        lengths = [len(item["input_ids"]) for item in batch]
        target_len = max(lengths) if lengths else 0
        batch_size = len(batch)
        input_ids = torch.full((batch_size, target_len), pad_id, dtype=torch.long)
        loss_mask = torch.zeros((batch_size, target_len), dtype=torch.long)
        for row, item in enumerate(batch):
            seq_len = len(item["input_ids"])
            if seq_len:
                input_ids[row, :seq_len] = torch.tensor(item["input_ids"], dtype=torch.long)
                loss_mask[row, :seq_len] = torch.tensor(item["loss_mask"], dtype=torch.long)
        labels = input_ids.clone()
        labels[loss_mask == 0] = -100
        return {"input_ids": input_ids, "labels": labels}

    num_train_epochs = args.tokens / total_tokens
    save_strategy = "no" if args.checkpoint_every <= 0 else "steps"

    training_args_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=per_device_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.grad_accum,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.num_workers,
        group_by_length=group_by_length,
        save_strategy=save_strategy,
        save_steps=args.checkpoint_every if save_strategy == "steps" else None,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": float(args.floor_factor)},
        save_total_limit=args.save_total_limit,
        report_to=["tensorboard"],
        disable_tqdm=True,
        optim="adamw_torch",
        fp16=(precision == "fp16"),
        bf16=(precision == "bf16"),
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
        dataloader_pin_memory=True,
    )
    training_args = TrainingArguments(**training_args_kwargs)

    class LoggingTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._log_start_time = None

        def log(self, logs, start_time=None):
            logs = dict(logs or {})
            logs["step"] = f"{self.state.global_step}/{self.state.max_steps}"
            now = time.time()
            if start_time is None:
                if self._log_start_time is None:
                    self._log_start_time = now
                start_time = self._log_start_time
            elapsed = now - start_time
            logs["elapsed_sec"] = int(elapsed)
            if self.state.global_step > 0:
                remaining = elapsed * (self.state.max_steps - self.state.global_step) / self.state.global_step
                logs["eta_sec"] = int(remaining)
            for key, value in list(logs.items()):
                if isinstance(value, float):
                    logs[key] = round(value, 5)
            return super().log(logs, start_time=start_time)

        def _get_train_sampler(self, train_dataset=None):
            if not self.args.group_by_length:
                return super()._get_train_sampler(train_dataset)
            train_dataset = train_dataset or self.train_dataset
            lengths = list(train_dataset[self.args.length_column_name])
            batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps

            class _LengthSampler(torch.utils.data.Sampler):
                def __init__(self, lengths, batch_size):
                    self.lengths = lengths
                    self.batch_size = batch_size

                def __len__(self):
                    return len(self.lengths)

                def __iter__(self):
                    indices = get_length_grouped_indices(
                        self.lengths, self.batch_size, mega_batch_mult=1
                    )
                    return iter(indices)

            return _LengthSampler(lengths, batch_size)

    trainer = LoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model()


if __name__ == "__main__":
    main()
