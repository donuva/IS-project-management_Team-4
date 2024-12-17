import re
import os
import argparse
import nltk
from nltk.translate.bleu_score import sentence_bleu
nltk.download('punkt')
nltk.download('punkt_tab')
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import peft
import pandas as pd
from datasets import Dataset

def get_image_list(image_folder_path, doc_file):
    with open(doc_file, 'r') as doc:
        article_list = doc.readlines()

    clean_ar_list = [re.sub(r'\.txt|\n', '', article)
                     for article in article_list]

    image_list = []
    for article in clean_ar_list:
        article_images = [
            re.sub(r'\.jpg', '', img)
            for img in os.listdir(image_folder_path)
            if article in img
        ]
        # append images or 'None' if no images found
        image_list.append(article_images if article_images else 'None')

    return image_list


def read_captions_from_list(caption_list, caption_dir):
    all_captions = []
    all_captions_file_name = []
    for captions_image in caption_list:
        if captions_image == 'None':  # skip articles with no captions
            continue

        for file_path in captions_image:
            caption_path = os.path.join(caption_dir, file_path)
            if not os.path.exists(caption_path):  # skip non-existent files
                continue

            with open(caption_path, "r") as f:
                captions = f.readlines()
                for cap in captions:
                    file_name_without_ext = os.path.splitext(file_path)[0]
                    all_captions_file_name.append(file_name_without_ext)
                    all_captions.append(cap.strip())

    return all_captions_file_name, all_captions

def get_model_predictions(model_dir, all_article_file):
    print("Loading saved model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    lora_model = peft.get_peft_model(
        model, peft.LoraConfig(r=16, lora_alpha=32))
    lora_model.eval() 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_model = lora_model.to(device)

    with open(all_article_file, 'r') as f:
        articles = f.readlines()  
    articles = [line.strip() for line in articles]

    data = {'article': []}
    for article in articles:  
        data['article'].append(article)
    dataset = Dataset.from_dict(data)

    all_predictions = []
    for i, example in enumerate(dataset):
        article = example["article"]

        input_ids = tokenizer.encode(
            article, return_tensors="pt", max_length=512, truncation=True).to(device)
        output_ids = lora_model.generate(
            input_ids, max_length=128, num_beams=4, early_stopping=True)
        generated_summary = tokenizer.decode(
            output_ids[0], skip_special_tokens=True)

        all_predictions.append(generated_summary)

    return all_predictions

def calculate_bleu_from_file(all_predictions, all_captions_file_name, all_captions):
    bleu_scores = []

    for prediction in all_predictions:
        target_sentence = prediction.strip()
        target_tokens = nltk.word_tokenize(target_sentence)

        for i in range(len(all_captions_file_name)):
            # Skip if the corresponding entry in all_captions is None
            if all_captions[i] == 'None':
                bleu_scores.append(None)
                continue

            reference_caption = all_captions[i] 
            reference_tokens = nltk.word_tokenize(reference_caption)
            score = sentence_bleu([reference_tokens], target_tokens)
            bleu_scores.append(score)

    return bleu_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write images for articles to file.")
    parser.add_argument("-m", "--model_dir", type=str, default="./saved_model",
                        help="Directory of the saved model.")
    parser.add_argument("-d", "--doc_file", required=True,
                        help="Name of file containing all article files names (Ex: test_doc_name.txt)")
    parser.add_argument("-a", "--article_file",
                        required=True, help="Name of file containing all articles (Ex: test_document.txt)")
    parser.add_argument("-i", "--image_folder_path",
                        required=True, help="Path to the image folder")
    parser.add_argument("-c", "--caption_folder_path",
                        required=True, help="Path to the caption folder")
    parser.add_argument("--type", required=True, choices=["train", "test", "valid"], help="Specify if this is for training or testing")
    args = parser.parse_args()

    # get the image list for articles
    print('Getting image list...')
    image_list = get_image_list(args.image_folder_path, args.doc_file)

    # sort image lists
    print('Sorting image list...')
    sorted_image_list = []
    for sublist in image_list:
        if sublist == 'None':
            sorted_image_list.append('None')
        else:
            sorted_image_list.append(
                sorted(sublist, key=lambda x: int(
                    re.search(r'_(\d+)$', x).group(1)))
            )

    # prepare captions
    caption_list = []
    for image_names in sorted_image_list:
        if image_names == 'None':
            caption_list.append('None')
        else:
            temp = [f"{image_name}.caption" for image_name in image_names]
            caption_list.append(temp)

    # read all captions
    all_captions_file_name, all_captions = read_captions_from_list(
        caption_list, args.caption_folder_path)
    print(f"Total caption file names: {len(all_captions_file_name)}")
    print(f"Total captions: {len(all_captions)}")

    # write all captions to file
    print('Writing captions to file...')
    with open("all_captions.txt", "w") as output_file:
        for file_name, caption in zip(all_captions_file_name, all_captions):
            output_file.write(file_name + " " + caption + "\n")

    # get model predictions
    all_predictions = get_model_predictions(args.model_dir, args.article_file)
    # calculate BLEU scores
    print('Calculating...')
    bleu_scores = calculate_bleu_from_file(all_predictions, all_captions_file_name, all_captions)

    # write BLEU scores to file
    print('Writing BLEU scores to file...')
    bleu_scores_filename = f"bleu_scores_captions_{args.type}.txt"

    # write bleu scores to file
    end_file_name = "end_file_name"
    with open(bleu_scores_filename, "w") as f:
        for file_name, score in zip(all_captions_file_name, bleu_scores):
            f.write(f"{file_name} BLEU={score}\n")
        f.write(f"{end_file_name} BLEU={0.0}\n")

    print(f"BLEU scores saved to {bleu_scores_filename}")
    print("Done.")
