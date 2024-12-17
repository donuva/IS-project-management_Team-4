import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import peft
import pandas as pd
from datasets import Dataset
import evaluate
from torch.utils.data import DataLoader


def load_dataset(file_path):
    data = pd.read_csv(file_path)
    return Dataset.from_pandas(data)


def evaluate_metrics(predictions, references):
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    # Loại bỏ các giá trị rỗng hoặc None
    valid_data = [(pred, ref) for pred, ref in zip(predictions, references) if pred and ref]
    if not valid_data:
        raise ValueError("No valid predictions and references for evaluation.")

    predictions, references = zip(*valid_data)

    # Tính ROUGE
    rouge_result = rouge.compute(predictions=list(predictions), references=list(references))

    bleu_result = bleu.compute(predictions=predictions, references=references)

    return rouge_result, bleu_result


def collate_fn(batch, tokenizer, device):
    """
    Hàm kết hợp batch, chuẩn bị đầu vào và tham chiếu.
    """
    articles = [item["article"] for item in batch]
    references = [item["highlights"] for item in batch]
    inputs = tokenizer(
        articles,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=True
    ).to(device)
    return inputs, references


def main(args):
    print("Loading saved model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    lora_model = peft.get_peft_model(
        model, peft.LoraConfig(r=16, lora_alpha=32))
    lora_model.eval()  # Set model to evaluation mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_model = lora_model.to(device)

    # Load test dataset
    test_dataset = load_dataset("../../nhom4/data/cnndailymail/test.csv")
    if args.num_records > 0:
        test_dataset = test_dataset.select(
            range(min(args.num_records, len(test_dataset))))

    print(f"Testing on {len(test_dataset)} records...")

    # Tạo DataLoader với batch_size
    batch_size = args.batch_size
    data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, device)
    )

    predictions = []
    references = []

    for step, (inputs, batch_references) in enumerate(data_loader):
        output_ids = lora_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
        batch_predictions = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)

        predictions.extend(batch_predictions)
        references.extend(batch_references)

    # Evaluate metrics
    try:
        rouge_result, bleu_result = evaluate_metrics(predictions, references)

        print("\nEvaluation Results:")
        print(f"ROUGE: {rouge_result}")
        print(f"BLEU: {bleu_result}")
    except ValueError as e:
        print(f"\nEvaluation Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a saved summarization model.")
    parser.add_argument(
        "-m", "--model_dir", type=str, default="./saved_model",
        help="Directory of the saved model."
    )
    parser.add_argument(
        "-n", "--num_records", type=int, default=-1,
        help="Number of test records to process. Use -1 for all records."
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=64,
        help="Batch size for testing."
    )
    args = parser.parse_args()
    main(args)

