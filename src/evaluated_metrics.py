import argparse
from os import read
from rouge_score.rouge_scorer import re
import torch
import pandas as pd
import evaluate

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def evaluate_metrics(predictions, references):
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    # Loại bỏ các giá trị rỗng hoặc None
    valid_data = [(pred, ref) for pred, ref in zip(predictions, references) if pred and ref]
    if not valid_data:
        raise ValueError("No valid predictions and references for evaluation.")

    predictions, references = zip(*valid_data)

    rouge_result = rouge.compute(predictions=list(predictions), references=list(references))
    bleu_result = bleu.compute(predictions=predictions, references=references)

    return rouge_result, bleu_result


def main(args):
    predictions = read_file(args.predict_file)
    references = read_file(args.target_file)

    # Evaluate metrics
    try:
        rouge_result, bleu_result = evaluate_metrics(predictions, references)

        print("\nEvaluation Results:")
        print(f"ROUGE: {rouge_result}")
        print(f"BLEU: {bleu_result}")
    except ValueError as e:
        print(f"\nEvaluation Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a saved summarization model.")
    parser.add_argument(
        "-predict", "--predict_file", required=True,
        help="Path to the file containing all articles (Ex: test_documents.txt)."
    )
    parser.add_argument(
        "-target", "--target_file", required=True,
        help="Path to the file containing all target summaries (Ex: tgt_test.txt)."
    )
    parser.add_argument(
        "-n", "--num_records", type=int, default=-1,
        help="Number of test records to process. Use -1 for all records."
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=16,
        help="Batch size for testing."
    )
    args = parser.parse_args()
    main(args)
