import re
import os
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from tqdm import tqdm
import argparse

contraction_mapping = {"ain’t": "is not", "aren’t": "are not", "can’t": "cannot", "’cause": "because", "could’ve": "could have", "couldn’t": "could not",
                       "didn’t": "did not", "doesn’t": "does not", "don’t": "do not", "hadn’t": "had not", "hasn’t": "has not", "haven’t": "have not",
                       "he’d": "he would", "he’ll": "he will", "he’s": "he is", "how’d": "how did", "how’d’y": "how do you", "how’ll": "how will", "how’s": "how is",
                       "I’d": "I would", "I’d’ve": "I would have", "I’ll": "I will", "I’ll’ve": "I will have", "I’m": "I am", "I’ve": "I have", "i’d": "i would",
                       "i’d’ve": "i would have", "i’ll": "i will",  "i’ll’ve": "i will have", "i’m": "i am", "i’ve": "i have", "isn’t": "is not", "it’d": "it would",
                       "it’d’ve": "it would have", "it’ll": "it will", "it’ll’ve": "it will have", "it’s": "it is", "let’s": "let us", "ma’am": "madam",
                       "mayn’t": "may not", "might’ve": "might have", "mightn’t": "might not", "mightn’t’ve": "might not have", "must’ve": "must have",
                       "mustn’t": "must not", "mustn’t’ve": "must not have", "needn’t": "need not", "needn’t’ve": "need not have", "o’clock": "of the clock",
                       "oughtn’t": "ought not", "oughtn’t’ve": "ought not have", "shan’t": "shall not", "sha’n’t": "shall not", "shan’t’ve": "shall not have",
                       "she’d": "she would", "she’d’ve": "she would have", "she’ll": "she will", "she’ll’ve": "she will have", "she’s": "she is",
                       "should’ve": "should have", "shouldn’t": "should not", "shouldn’t’ve": "should not have", "so’ve": "so have", "so’s": "so as",
                       "this’s": "this is", "that’d": "that would", "that’d’ve": "that would have", "that’s": "that is", "there’d": "there would",
                       "there’d’ve": "there would have", "there’s": "there is", "here’s": "here is", "they’d": "they would", "they’d’ve": "they would have",
                       "they’ll": "they will", "they’ll’ve": "they will have", "they’re": "they are", "they’ve": "they have", "to’ve": "to have",
                       "wasn’t": "was not", "we’d": "we would", "we’d’ve": "we would have", "we’ll": "we will", "we’ll’ve": "we will have", "we’re": "we are",
                       "we’ve": "we have", "weren’t": "were not", "what’ll": "what will", "what’ll’ve": "what will have", "what’re": "what are",
                       "what’s": "what is", "what’ve": "what have", "when’s": "when is", "when’ve": "when have", "where’d": "where did", "where’s": "where is",
                       "where’ve": "where have", "who’ll": "who will", "who’ll’ve": "who will have", "who’s": "who is", "who’ve": "who have",
                       "why’s": "why is", "why’ve": "why have", "will’ve": "will have", "won’t": "will not", "won’t’ve": "will not have",
                       "would’ve": "would have", "wouldn’t": "would not", "wouldn’t’ve": "would not have", "y’all": "you all",
                       "y’all’d": "you all would", "y’all’d’ve": "you all would have", "y’all’re": "you all are", "y’all’ve": "you all have",
                       "you’d": "you would", "you’d’ve": "you would have", "you’ll": "you will", "you’ll’ve": "you will have",
                       "you’re": "you are", "you’ve": "you have", "ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                       "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                       "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would",
                       "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                       "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
                       "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
                       "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
                       "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                       "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
                       "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                       "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would", "they'd've": "they would have",
                       "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                       "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                       "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                       "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                       "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                       "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                       "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                       "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                       "you're": "you are", "you've": "you have", "n't": 'not'}


def cleaning(string, contraction_mapping):
    low = str(string).lower()
    soup = BeautifulSoup(low, 'lxml')
    low = soup.text
    low = re.sub(" '", "'", low)
    low = re.sub(" n't", "n't", low)
    sent = []
    for m in low.split():
        if m in contraction_mapping:
            sent.append(contraction_mapping[m])
        else:
            sent.append(m)
    jnt = ' '.join(sent)
    low = re.sub("'s", "", jnt)
    low = re.sub("’s", "", low)
    new = re.sub("\n", '', low)
    new = re.sub(r'[\$\"\(\)\)\#\:\@\=\>\<\-\`\-\-\/\;\‘\£\%\*\—]', ' ', new)
    new = re.sub(",", '', new)
    new = re.sub('\!', '.', new)
    new = re.sub('\?', '.', new)
    new = re.sub("'", '', new)
    new = re.sub("°", '', new)
    new = re.sub("\.\.\.", '.', new)
    new = re.sub(r"[^a-zA-Z0-9]", ' ', new)
    new = (re.sub(r'[\s]+', ' ', new)).strip()
    return new


def process_captions(type, contraction_mapping, img_dir, caption_dir):
    with open(f"{type}_doc_name.txt", 'r') as doc:
        doc_list = doc.readlines()
    doc = []
    for i in doc_list:
        doc.append(re.sub('\n', '', i))
    # removing .txt extention to search the image names related to the article
    doc_clean = [re.sub('\.txt', '', i) for i in doc]

    # doc_clean contains article file names
    # img contains all image names
    img_list = []   # This will be containing image names related to every article
    img = os.listdir(img_dir)
    for i in tqdm(doc_clean):       # Looping through every article file name
        art_img_list = []
        for j in range(len(img)):   # Looping through every image name
            if i in img[j]:
                art_img_list.append(re.sub('\.jpg', '', img[j].strip()))
        if art_img_list == []:
            img_list.append('None')
        else:
            img_list.append(art_img_list)

    # Saving image file names in .txt file
    if type == "train":
        name = "Train"
    elif type == "test":
        name = "Test"
    elif type == "valid":
        name = "Valid"
    with open(f"{name}_Image_Names.txt", 'w') as file:
        for img_names in img_list:
            if img_names != 'None':
                file.write('\t'.join(img_names) + '\n')

    summary = []
    for i in tqdm(img_list):
        img_summ = []
        for img in i:
            caption_path = os.path.join(caption_dir, img + '.caption')
            print(f"Checking caption file: {caption_path}")

            if os.path.exists(caption_path):  # Check if the file exists
                try:
                    with open(caption_path, 'r', encoding='utf-8') as file:
                        caption = file.readlines()
                        cap = cleaning(caption[0] if caption else '', contraction_mapping)
                        img_summ.append(cap if cap.strip() else "None")
                except Exception as e:
                    print(f"Error reading file {caption_path}: {e}")
                    img_summ.append("Error")
            else:
                print(f"Warning: Caption file not found: {caption_path}")
                img_summ.append("None")

        summary.append(img_summ)

    with open(f"{name}_Image_Captions.txt", 'w') as file:
        for i in summary:
            file.write('\t'.join(i))
            file.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean captions from articles.")
    parser.add_argument("-i", "--img_dir", required=True,
                        help="Path to the image directory")
    parser.add_argument("-c", "--caption_dir", required=True,
                        help="Path to the caption directory")
    parser.add_argument("--type", required=True, choices=[
                        "train", "test", "valid"], help="Specify if this is for training or testing")
    args = parser.parse_args()
    print("Processing...")
    process_captions(args.type, contraction_mapping,
                     args.img_dir, args.caption_dir)
    print("Done.")