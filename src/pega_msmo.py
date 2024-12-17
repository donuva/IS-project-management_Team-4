import argparse
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, TaskType, get_peft_model
import numpy as np
from datasets import Dataset


def load_dataset(all_article_file, target_file):
    with open(all_article_file, "r") as f:
        articles = f.readlines()
    articles = [line.strip() for line in articles]

    with open(target_file, "r") as f:
        targets = f.readlines()
    targets = [line.strip() for line in targets]

    data = {'article': articles, 'highlights': targets}
    return Dataset.from_dict(data)


def preprocess_data(examples, tokenizer):
    inputs = examples["article"]
    targets = examples["highlights"]

    # Token hóa input
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length",
    )

    # Token hóa target
    labels = tokenizer(
        targets,
        max_length=128,
        truncation=True,
        padding="max_length",
    )["input_ids"]

    # Thay padding token thành -100
    labels = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels]
    model_inputs["labels"] = labels
    return model_inputs


def main(args):
    print(f"Loading summarization model: {args.model}")

    # Load tokenizer và model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Collator dữ liệu
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Load dataset
    train_dataset = load_dataset(args.all_train_article_file, args.train_target_file)
    if args.num_records > 0:
        train_dataset = train_dataset.select(range(args.num_records))
    train_dataset = train_dataset.map(lambda x: preprocess_data(x, tokenizer), batched=True)

    # Cấu hình LoRA
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["encoder.layers", "decoder.layers"],
        bias="none",
        lora_dropout=0.05,
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    lora_model = get_peft_model(model, lora_config)
    lora_model = lora_model.to(device)

    # Cấu hình huấn luyện
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        gradient_accumulation_steps=4,
        num_train_epochs=3,  # Số epoch để kiểm tra
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        fp16=True,
        save_total_limit=1,
        save_steps=500,
        report_to="none",
    )

    # Tạo Trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Debug loss và gradient
    def debug_training_step():
        lora_model.train()
        optimizer = torch.optim.AdamW(lora_model.parameters(), lr=5e-5)
        for step, batch in enumerate(trainer.get_train_dataloader()):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = lora_model(**batch)
            loss = outputs.loss

            if torch.isnan(loss):
                print("NaN loss detected at step:", step)
                return

            loss.backward()

            # Clip gradient để tránh NaN
            grad_norm = clip_grad_norm_(lora_model.parameters(), max_norm=1.0)
            print(f"Step: {step}, Loss: {loss.item()}, Grad Norm: {grad_norm}")

            optimizer.step()
            optimizer.zero_grad()
            if step > 10:  # Giới hạn số bước debug
                break

    print("Debugging training step...")
    debug_training_step()

    print("Starting training...")
    trainer.train()
    print("Training complete!")

    # Lưu mô hình
    print("Saving the trained model...")
    lora_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a summarization model.")
    parser.add_argument(
        "-m", "--model", type=str, default="google/pegasus-cnn_dailymail",
        help="Name of the pre-trained summarization model to use."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="./saved_model_pegasus",
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
    args = parser.parse_args()
    main(args)

