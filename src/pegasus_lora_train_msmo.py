import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
import peft
from peft import TaskType
import pandas as pd
from datasets import Dataset
from accelerate import Accelerator

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
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    # Add labels to model inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Flatten predictions and labels for cross-entropy calculation
    predictions = np.argmax(logits, axis=-1)
    labels = np.where(labels != -100, labels, -1)  # Mask ignored tokens (-100)

    # Calculate loss (if needed, using a custom loss function or direct logit comparison)
    loss = torch.nn.CrossEntropyLoss()(torch.tensor(logits), torch.tensor(labels))
    return {"eval_loss": loss.item()}

def main(args):
    print(f"Loading summarization model: {args.model}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model,quantization_config=bnb_config)
    # model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Load datasets
    train_dataset = load_dataset(args.all_train_article_file, args.train_target_file)
    if args.num_records > 0:
        train_dataset = train_dataset.select(range(args.num_records))
    
    # Preprocess datasets
    train_dataset = train_dataset.map(
        lambda x: preprocess_data(x, tokenizer), batched=True)
    
    # Configure LoRA
    lora_config = peft.LoraConfig(
        r=32, #Rank
        lora_alpha=32,
        target_modules=['q_proj',
                        'v_proj',],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type=TaskType.SEQ_2_SEQ_LM   #pegasus
    )

    lora_model = peft.get_peft_model(model, lora_config)
    lora_model = lora_model.to(device)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=5,
        fp16=True,
        # evaluation_strategy="epoch",
        # save_strategy="epoch",
        # save_total_limit=2,
        # metric_for_best_model="eval_loss",
        # load_best_model_at_end=True,
        report_to="none",
    )

    accelerator = Accelerator()
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        #accelerator=accelerator,
        #compute_metrics=compute_metrics,
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
