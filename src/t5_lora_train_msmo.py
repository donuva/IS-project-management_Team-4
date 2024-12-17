
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
import peft
import pandas as pd
from datasets import Dataset


def load_dataset(all_article_file, target_file):
    with open(all_article_file, "r") as f:
        articles = f.readlines()  
    articles = [line.strip() for line in articles]
    #print("Articles in _doc_name.txt: ", articles)

    with open(target_file, "r") as f:
        targets = f.readlines()
    targets = [line.strip() for line in targets]
    #print("Target in tgt_.txt: ", targets)

    data = {'article': [], 'highlights': []}
    for article, target in zip(articles, targets):  
        data['article'].append(article)
        data['highlights'].append(target)

    return Dataset.from_dict(data)


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
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    # Add labels to model inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(p):
    # Tính toán eval_loss
    preds, labels = p
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)  
    loss = loss_fct(torch.tensor(preds), torch.tensor(labels))
    return {"eval_loss": loss.item()}


def main(args):
    print(f"Loading summarization model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Load datasets
    train_dataset = load_dataset(args.all_train_article_file, args.train_target_file)
    if args.num_records > 0:
        train_dataset = train_dataset.select(range(args.num_records))
    val_dataset = load_dataset(args.all_valid_article_file, args.valid_target_file)
    if args.num_records > 0:
        val_dataset = val_dataset.select(range(args.num_records))

    # Preprocess datasets
    train_dataset = train_dataset.map(
        lambda x: preprocess_data(x, tokenizer), batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_data(x, tokenizer), batched=True
    )

    # Configure LoRA
    lora_config = peft.LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["encoder.block.0.layer.0.SelfAttention.q",
                        "encoder.block.0.layer.0.SelfAttention.v"],
        lora_dropout=0.1,
    )
    lora_model = peft.get_peft_model(model, lora_config)
    lora_model = lora_model.to(device)

    training_args = TrainingArguments(
        per_device_train_batch_size=16,  # Batch size for training
        per_device_eval_batch_size=16,  # Batch size for evaluation
        output_dir="./results",  # Directory to save model checkpoints
        # num_train_epochs=1,  # Number of training epochs
        # learning_rate=5e-5,  # Learning rate for optimization
        # weight_decay=0.01,  # Weight decay for regularization
        # logging_dir="./content/logs",  # Directory for logging
        # logging_steps=10,  # Log every 10 steps
        # evaluation_strategy="epoch",  # Evaluate after every epoch
        # save_strategy="epoch",  # Save model after every epoch
        # save_total_limit=2,  # Keep only the latest 2 checkpoints
        # metric_for_best_model="eval_loss",  # Metric to select the best model
        # load_best_model_at_end=True,  # Load the best model after training
        # report_to="none",  # Disable reporting to external tools
    )


    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")

    # Save the model and tokenizer
    print("Saving the trained model...")
    lora_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a summarization model.")
    parser.add_argument(
        "-m", "--model", type=str, default="t5-small",
        help="Name of the pre-trained summarization model to use."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="./saved_model",
        help="Directory to save the trained model."
    )
    parser.add_argument(
        "-n", "--num_records", type=int, default=-1,
        help="Number of test records to process. Use -1 for all records."
    )
    parser.add_argument(
        "-atr", "--all_train_article_file", required=True,
        help="Path to the file containing all articles (Ex: train_document.txt)."
    )
    parser.add_argument(
        "-ttr", "--train_target_file", required=True,
        help="Path to the file containing all target summaries (Ex: tgt_train.txt)."
    )
    parser.add_argument(
        "-av", "--all_valid_article_file", required=True,
        help="Path to the file containing all article files names (Ex: valid_document.txt)."
    )
    parser.add_argument(
        "-tv", "--valid_target_file", required=True,
        help="Path to the file containing all target summaries (Ex: valid_test.txt)."
    )
    args = parser.parse_args()
    main(args)
