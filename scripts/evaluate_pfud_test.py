import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


DEFAULT_DATASET = Path(r"C:\Users\12240\Desktop\ssh\protdata\PFUD_replaced_with_neighbors_filtered_top1ge20.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a local HF model (Qwen/Llama and optional LoRA) on PFUD test split."
    )
    parser.add_argument("--model-path", required=True, help="Base model path or HF model id.")
    parser.add_argument("--lora-path", default=None, help="Optional LoRA adapter path exported by LLaMA-Factory.")
    parser.add_argument(
        "--dataset-path",
        default=str(DEFAULT_DATASET),
        help="Path to the dataset JSON file. Defaults to PFUD_replaced_with_neighbors_filtered_top1ge20.json.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Where to save prediction results as JSONL.",
    )
    parser.add_argument(
        "--metrics-path",
        default=None,
        help="Optional path to save summary metrics as JSON.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate. Default: test",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional max number of samples to run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Alias of --max-samples. If both are set, the smaller value is used.",
    )
    parser.add_argument(
        "--batch-size",
        "--bathc-size",
        dest="batch_size",
        type=int,
        default=1,
        help="Batch size for generation. '--bathc-size' is kept as a compatibility alias.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens per generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p for sampling when temperature > 0.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cuda, cpu, cuda:0, etc.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading tokenizer/model.",
    )
    parser.add_argument(
        "--disable-chat-template",
        action="store_true",
        help="If set, build a plain prompt instead of using tokenizer chat template.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a short log line for each sample.",
    )
    return parser.parse_args()


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array.")
    return data


def normalize_split(split: Optional[str]) -> str:
    if split == "validation":
        return "valid"
    return split or ""


def select_samples(
    data: List[Dict[str, Any]],
    split: str,
    max_samples: Optional[int],
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for row in data:
        if normalize_split(row.get("split")) != split:
            continue
        selected.append(row)
        if max_samples is not None and len(selected) >= max_samples:
            break
    return selected


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_for_metric(text: str) -> List[str]:
    return normalize_text(text).split()


def exact_match(pred: str, ref: str) -> float:
    return float(normalize_text(pred) == normalize_text(ref))


def token_f1(pred: str, ref: str) -> float:
    pred_tokens = tokenize_for_metric(pred)
    ref_tokens = tokenize_for_metric(ref)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    ref_counts: Dict[str, int] = {}
    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1

    common = 0
    for tok in pred_tokens:
        if ref_counts.get(tok, 0) > 0:
            common += 1
            ref_counts[tok] -= 1

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def char_overlap_ratio(pred: str, ref: str) -> float:
    pred_chars = normalize_text(pred)
    ref_chars = normalize_text(ref)
    if not pred_chars and not ref_chars:
        return 1.0
    if not pred_chars or not ref_chars:
        return 0.0
    ref_counts: Dict[str, int] = {}
    for ch in ref_chars:
        ref_counts[ch] = ref_counts.get(ch, 0) + 1
    common = 0
    for ch in pred_chars:
        if ref_counts.get(ch, 0) > 0:
            common += 1
            ref_counts[ch] -= 1
    return common / max(len(ref_chars), 1)


def resolve_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return "auto"


def resolve_device_map(device: str):
    if device == "auto":
        return "auto"
    if device.startswith("cuda"):
        return {"": device}
    if device == "cpu":
        return {"": "cpu"}
    return {"": device}


def build_prompt(
    tokenizer: AutoTokenizer,
    instruction: str,
    input_text: str,
    disable_chat_template: bool,
) -> str:
    user_text = instruction.strip()
    if input_text:
        user_text = f"{user_text}\n\n{input_text.strip()}"

    if disable_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return f"User: {user_text}\nAssistant:"

    messages = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"User: {user_text}\nAssistant:"


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=resolve_device_map(args.device),
        trust_remote_code=args.trust_remote_code,
    )

    if args.lora_path:
        if PeftModel is None:
            raise ImportError("peft is required for --lora-path, but it is not installed.")
        model = PeftModel.from_pretrained(model, args.lora_path)

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[Tuple[str, int]]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    outputs = model.generate(**inputs, **generation_kwargs)
    results: List[Tuple[str, int]] = []
    for i, prompt_len in enumerate(prompt_lens):
        gen_ids = outputs[i][int(prompt_len):]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        results.append((text, len(gen_ids)))
    return results


def summarize_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    total = len(rows)
    if total == 0:
        return {
            "num_samples": 0,
            "exact_match": 0.0,
            "token_f1": 0.0,
            "char_overlap_ratio": 0.0,
            "avg_generated_tokens": 0.0,
        }

    em_sum = 0.0
    f1_sum = 0.0
    overlap_sum = 0.0
    gen_tokens_sum = 0
    for row in rows:
        em_sum += row["metrics"]["exact_match"]
        f1_sum += row["metrics"]["token_f1"]
        overlap_sum += row["metrics"]["char_overlap_ratio"]
        gen_tokens_sum += row["generated_tokens"]

    return {
        "num_samples": total,
        "exact_match": em_sum / total,
        "token_f1": f1_sum / total,
        "char_overlap_ratio": overlap_sum / total,
        "avg_generated_tokens": gen_tokens_sum / total,
    }


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)
    metrics_path = Path(args.metrics_path) if args.metrics_path else output_path.with_suffix(".metrics.json")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    limit = args.max_samples
    if args.limit is not None:
        limit = args.limit if limit is None else min(limit, args.limit)

    data = load_json_array(dataset_path)
    samples = select_samples(data, split=args.split, max_samples=limit)
    if not samples:
        raise ValueError(f"No samples found for split={args.split!r} in {dataset_path}")

    model, tokenizer = load_model_and_tokenizer(args)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as fout:
        for batch_start in range(0, len(samples), args.batch_size):
            batch_samples = samples[batch_start:batch_start + args.batch_size]
            prompts = [
                build_prompt(
                    tokenizer=tokenizer,
                    instruction=sample.get("instruction", ""),
                    input_text=sample.get("input", ""),
                    disable_chat_template=args.disable_chat_template,
                )
                for sample in batch_samples
            ]
            predictions = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            for offset, (sample, (prediction, generated_tokens)) in enumerate(zip(batch_samples, predictions)):
                idx = batch_start + offset + 1
                reference = sample.get("output", "")
                row = {
                    "index": idx - 1,
                    "accesion": sample.get("accesion"),
                    "split": sample.get("split"),
                    "instruction": sample.get("instruction", ""),
                    "input": sample.get("input", ""),
                    "reference": reference,
                    "prediction": prediction,
                    "generated_tokens": generated_tokens,
                    "metrics": {
                        "exact_match": exact_match(prediction, reference),
                        "token_f1": token_f1(prediction, reference),
                        "char_overlap_ratio": char_overlap_ratio(prediction, reference),
                    },
                }
                results.append(row)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

                if args.verbose:
                    print(
                        f"[{idx}/{len(samples)}] accesion={row['accesion']} "
                        f"token_f1={row['metrics']['token_f1']:.4f} "
                        f"gen_tokens={generated_tokens}"
                    )

    summary = {
        "model_path": args.model_path,
        "lora_path": args.lora_path,
        "dataset_path": str(dataset_path),
        "split": args.split,
        "max_samples": limit,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "metrics": summarize_metrics(results),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
