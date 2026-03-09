import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm
from vllm import LLM, SamplingParams


DEFAULT_DATASET = Path(r"C:\Users\12240\Desktop\ssh\protdata\PFUD_replaced_with_neighbors_filtered_top1ge20.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a model using vLLM on PFUD test split."
    )
    parser.add_argument("--model-path", required=True, help="Model path or HF model id.")
    parser.add_argument(
        "--lora-path",
        default=None,
        help="Path to LoRA adapter (trained by LlamaFactory).",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(DEFAULT_DATASET),
        help="Path to the dataset JSON file.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Where to save prediction results as JSONL.",
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
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=5120,
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
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode for Qwen models (passes enable_thinking=True).",
    )
    return parser.parse_args()


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array.")
    return data


def build_prompt(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build chat messages for vLLM."""
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")

    # Combine instruction and input as user message
    user_message = f"{instruction}\n\n{input_text}"

    return [{"role": "user", "content": user_message}]


def main():
    args = parse_args()

    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_path}")
    data = load_json_array(dataset_path)

    samples = [s for s in data if s.get("split") == args.split]
    print(f"Found {len(samples)} samples in split '{args.split}'")

    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Limited to {len(samples)} samples")

    print(f"Loading model from {args.model_path}")

    # Configure LoRA if provided
    enable_lora = args.lora_path is not None

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enable_lora=enable_lora,
        max_lora_rank=64 if enable_lora else None,
    )

    if args.lora_path:
        print(f"Loading LoRA adapter from {args.lora_path}")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    print(f"Starting evaluation with batch_size={args.batch_size}")

    tokenizer = llm.get_tokenizer()

    with output_path.open("w", encoding="utf-8") as fout:
        for batch_start in tqdm(range(0, len(samples), args.batch_size), desc="Evaluating", unit="batch"):
            batch_samples = samples[batch_start:batch_start + args.batch_size]

            # Build chat messages and apply tokenizer's chat template
            prompts = []
            for s in batch_samples:
                messages = build_prompt(s)

                # Apply chat template with thinking mode disabled by default
                template_kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                    "enable_thinking": args.enable_thinking,
                }

                prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
                prompts.append(prompt)

            # Generate with LoRA if provided
            if args.lora_path:
                from vllm.lora.request import LoRARequest
                lora_request = LoRARequest("protein_lora", 1, args.lora_path)
                outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
            else:
                outputs = llm.generate(prompts, sampling_params)

            for sample, output in zip(batch_samples, outputs):
                prediction = output.outputs[0].text.strip()

                row = {
                    "accesion": sample.get("accesion", ""),
                    "split": sample.get("split", ""),
                    "instruction": sample.get("instruction", ""),
                    "input": sample.get("input", ""),
                    "prompt": output.prompt,
                    "prediction": prediction,
                    "reference": sample.get("output", ""),
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "model_path": args.model_path,
        "dataset_path": str(dataset_path),
        "split": args.split,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "total_samples": len(samples),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
