import os
import sys
import json
import re
print("Script started...", flush=True)
from statistics import mean
from bert_score import BERTScorer
from tqdm import tqdm
from transformers import AutoTokenizer
import evaluate
import torch

# Set environment variables for offline mode/mirrors if needed (copied from notebook)
os.environ["HF_HOME"] = "/remote-home/jxlei/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/remote-home/jxlei/.cache/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def extract_prediction_from_json(text):
    """Extract description from JSON-formatted function_predict field."""
    # Ensure input is a string
    if not isinstance(text, str):
        return str(text) if text else ""

    try:
        # Try to find JSON object in the text
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            json_obj = json.loads(json_match.group())
            desc = json_obj.get('description', text)
            return str(desc) if desc else text
    except:
        pass
    return text

def detect_field_names(entry):
    """
    Detect which field names to use for ground truth and prediction.
    Returns (gt_field, pred_field) tuple.

    Supported formats:
    1. ground_truth + predicted_function (prot_function_predictions.jsonl)
    2. label + retrieved_functions (filtered_retrieved_functions_and_labels.jsonl)
    3. label + function_predict (RAG_test.jsonl)
    4. function + predicted_function
    5. reference + prediction (evaluate_pfud_test_vllm.py output)
    """
    # Check for different field combinations
    if 'ground_truth' in entry and 'predicted_function' in entry:
        return 'ground_truth', 'predicted_function'
    elif 'label' in entry and 'retrieved_functions' in entry:
        return 'label', 'retrieved_functions'
    elif 'label' in entry and 'function_predict' in entry:
        return 'label', 'function_predict'
    elif 'function' in entry and 'predicted_function' in entry:
        return 'function', 'predicted_function'
    elif 'reference' in entry and 'prediction' in entry:
        return 'reference', 'prediction'
    else:
        return None, None

def main():
    # Accept input file from command line argument or use default
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "/remote-home/jxlei/protcot/prot_function_predictions.jsonl"

    print(f"Loading predictions from {input_file}...")

    # Check if file is JSONL or JSON format
    with open(input_file, 'r', encoding='utf-8') as f:
        # Try to detect format by file extension
        if input_file.endswith('.jsonl'):
            # JSONL format: one JSON object per line
            print("Detected JSONL format...")
            data = [json.loads(line) for line in f if line.strip()]
        else:
            # Regular JSON format
            print("Detected JSON format...")
            data = json.load(f)

    if not data:
        print("Error: No data loaded from file.")
        return

    # Auto-detect field names from first entry
    gt_field, pred_field = detect_field_names(data[0])
    if not gt_field or not pred_field:
        print(f"Error: Could not detect field names. First entry keys: {list(data[0].keys())}")
        return

    print(f"Detected fields: ground_truth='{gt_field}', prediction='{pred_field}'")

    ground_truths = []
    predictions = []

    # Initialize tokenizer for filtering
    model_path = "/remote-home/jxlei/models/biobert-large-cased-v1.1"
    print(f"Initializing tokenizer from {model_path} for length filtering...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Warning: Could not load tokenizer from {model_path}. Using default 'bert-base-cased' for approximation.")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    skipped_count = 0
    print("Filtering entries > 511 tokens...")

    for entry in tqdm(data, desc="Filtering"):
        gt = entry.get(gt_field)
        pred = entry.get(pred_field)

        # Special handling for function_predict field (may contain JSON)
        if pred_field == 'function_predict' and pred:
            pred = extract_prediction_from_json(pred)

        # Ensure both exist and are strings
        if gt and pred:
            # Convert to string if not already
            gt = str(gt) if not isinstance(gt, str) else gt
            pred = str(pred) if not isinstance(pred, str) else pred

            # Skip empty strings
            if not gt.strip() or not pred.strip():
                continue

            try:
                # Check token length
                # Note: BERT limit is 512. We check if > 511 to be safe (keeping margin for [CLS]/[SEP])
                len_gt = len(tokenizer.encode(gt, add_special_tokens=True, truncation=False))
                len_pred = len(tokenizer.encode(pred, add_special_tokens=True, truncation=False))

                if len_gt > 511 or len_pred > 511:
                    skipped_count += 1
                    continue

                ground_truths.append(gt)
                predictions.append(pred)
            except Exception as e:
                # Skip entries that cause encoding errors
                print(f"\nWarning: Skipping entry due to encoding error: {e}")
                skipped_count += 1
                continue

    print(f"Loaded {len(ground_truths)} records (skipped {skipped_count} due to length > 511).")

    # Generate output filename based on input filename
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"/remote-home/jxlei/protcot/f1_result_{input_basename}.txt"
    print(f"Results will be saved to: {output_file}")

    with open(output_file, "w") as out_f:
        out_f.write(f"Evaluation results for: {input_file}\n")
        out_f.write(f"Ground truth field: {gt_field}\n")
        out_f.write(f"Prediction field: {pred_field}\n")
        out_f.write(f"Total records: {len(ground_truths)}\n")
        out_f.write(f"Skipped records: {skipped_count}\n")
        out_f.write("="*60 + "\n\n")
        if len(ground_truths) == 0:
            print("No valid records found.")
            out_f.write("No valid records found.\n")
            return

        # --- Compute BLEU and ROUGE first (CPU based, fast) ---
        print("Computing BLEU and ROUGE...")
        out_f.write("Computing BLEU and ROUGE...\n")

        try:
            print("Attempting to load metrics via 'evaluate'...")
            bleu = evaluate.load("bleu")
            rouge = evaluate.load("rouge")

            # FIX: Wrap ground_truths in lists for proper reference format
            references = [[gt] for gt in ground_truths]

            bleu_results = bleu.compute(predictions=predictions, references=references)
            rouge_results = rouge.compute(predictions=predictions, references=references)

            # Output BLEU
            bleu_score = bleu_results['bleu']
            print(f"BLEU: {bleu_score:.4f}")
            out_f.write(f"BLEU: {bleu_score:.4f}\n")

            # Output ROUGE with better formatting
            print(f"ROUGE-1: {rouge_results['rouge1']:.4f}")
            print(f"ROUGE-2: {rouge_results['rouge2']:.4f}")
            print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")

            out_f.write(f"ROUGE-1: {rouge_results['rouge1']:.4f}\n")
            out_f.write(f"ROUGE-2: {rouge_results['rouge2']:.4f}\n")
            out_f.write(f"ROUGE-L: {rouge_results['rougeL']:.4f}\n")

        except Exception as e:
            err_msg = f"Error computing BLEU/ROUGE: {e}"
            print(err_msg)
            out_f.write(err_msg + "\n")
            import traceback
            traceback.print_exc()

        # --- Compute BERTScore F1 (GPU based, slower) ---
        print("\nComputing BERTScore F1...")
        out_f.write("\nComputing BERTScore F1...\n")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        model_path = "/remote-home/jxlei/models/biobert-large-cased-v1.1"

        try:
            scorer = BERTScorer(
                model_type=model_path,
                num_layers=24,
                rescale_with_baseline=False,
                device=device
            )
            print("Model loaded. Starting inference...")

            # FIX: Let BERTScorer handle batching internally for better efficiency
            # Or use manual batching without passing batch_size to score()
            batch_size = 64
            all_f1 = []

            for i in tqdm(range(0, len(ground_truths), batch_size), desc="Computing F1"):
                batch_preds = predictions[i:i+batch_size]
                batch_gts = ground_truths[i:i+batch_size]

                # FIX: Remove redundant batch_size parameter
                P, R, F1 = scorer.score(batch_preds, batch_gts, verbose=False)
                all_f1.extend([float(x) for x in F1])

            f1_mean = mean(all_f1)
            result_str = f"BERTScore F1 (mean): {f1_mean:.4f}"
            print(result_str)
            out_f.write(result_str + "\n")

            # Also output precision and recall for completeness
            print(f"Note: Individual P/R scores not saved, only F1 mean reported.")

        except Exception as e:
            err_msg = f"Error computing BERTScore: {e}"
            print(err_msg)
            out_f.write(err_msg + "\n")
            import traceback
            traceback.print_exc()
            print(f"Ensure that {model_path} exists and is a valid model directory.")

if __name__ == "__main__":
    main()
