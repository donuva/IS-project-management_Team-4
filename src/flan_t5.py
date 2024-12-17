import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq
import peft
import pandas as pd
from datasets import Dataset


def load_dataset(file_path):
    """
    Load a dataset from a CSV file and convert it into a Hugging Face Dataset object.
    """
    data = pd.read_csv(file_path)
    return Dataset.from_pandas(data)


def preprocess_data(examples, tokenizer):
    """
    Tokenize articles and highlights from the dataset with consistent padding and truncation.
    """
    inputs = examples["article"]
    targets = examples["highlights"]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Tokenize targets (labels)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    # Add labels to model inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main(args):
    # Load model and tokenizer
    print(f"Loading summarization model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        # padding=True,
        # truncation=True,
        # max_length=512,
    )

    train_dataset = load_dataset("../../nhom4/data/cnndailymail/train.csv")
    train_dataset = train_dataset.select(range(20000))
    val_dataset = load_dataset("../../nhom4/data/cnndailymail/validation.csv")

    # Preprocess datasets
    train_dataset = train_dataset.map(
        lambda x: preprocess_data(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(
        lambda x: preprocess_data(x, tokenizer), batched=True)

    lora_config = peft.LoraConfig(
        r=16,
        lora_alpha=32,
        # Check these modules for model compatibility
        target_modules=["encoder.block.0.layer.0.SelfAttention.q",
                        "encoder.block.0.layer.0.SelfAttention.v"],  # Adjust for T5 architecture
        # target_modules=["transformer.blocks.*.attn.Wqkv",  "transformer.blocks.*.attn.out_proj" ],
        lora_dropout=0.1,
    )
    lora_model = peft.get_peft_model(model, lora_config)
    lora_model = lora_model.to(device)

    # training_args = TrainingArguments(
    #     output_dir="./results",
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     learning_rate=5e-5,
    #     per_device_train_batch_size=4,
    #     per_device_eval_batch_size=4,
    #     num_train_epochs=3,
    #     weight_decay=0.01,
    #     logging_dir="./logs",
    #     logging_steps=10,
    #     metric_for_best_model="rouge1",
    #     save_total_limit=2,
    #     load_best_model_at_end=True,
    #     report_to="none",  # Disable reporting for simplicity
    # )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        # Calculate custom metrics
        result = metric.compute(predictions=decoded_preds,
                                references=decoded_labels)

        # Optionally add eval_loss if needed
        result["eval_loss"] = eval_pred.metrics.get("eval_loss", None)
        return result

    trainer = Trainer(
        model=lora_model,
        # args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")

    test_dataset = load_dataset("../../nhom4/data/cnndailymail/test.csv")

    if args.num_records > 0:
        test_dataset = test_dataset.select(
            range(min(args.num_records, len(test_dataset))))

    print(f"Testing on {len(test_dataset)} records...")
    for i, example in enumerate(test_dataset):
        article = example["article"]
        input_ids = tokenizer.encode(
            article, return_tensors="pt", max_length=512, truncation=True).to(device)
        output_ids = lora_model.generate(
            input_ids, max_length=128, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\nTest Record {i + 1}:")
        print(f"Article: {article}")
        print(f"Generated Summary: {summary}")
        print(f"Reference Summary: {example['highlights']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a summarization model and test it.")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="google/flan-t5-small",
        help="Name of the pre-trained summarization model to use (e.g., facebook/bart-large-cnn, t5-small)."
    )
    parser.add_argument(
        "-n",
        "--num_records",
        type=int,
        default=-1,
        help="Number of test records to process. Use -1 to process the entire test dataset."
    )

    args = parser.parse_args()
    main(args)
