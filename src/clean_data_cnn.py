import argparse
import re
import os
from tqdm import tqdm
from bs4 import BeautifulSoup

contraction_mapping = {"ain’t": "is not", "aren’t": "are not","can’t": "cannot", "’cause": "because", "could’ve": "could have", "couldn’t": "could not",
                          "didn’t": "did not", "doesn’t": "does not", "don’t": "do not", "hadn’t": "had not", "hasn’t": "has not", "haven’t": "have not",
                          "he’d": "he would","he’ll": "he will", "he’s": "he is", "how’d": "how did", "how’d’y": "how do you", "how’ll": "how will", "how’s": "how is",
                          "I’d": "I would", "I’d’ve": "I would have", "I’ll": "I will", "I’ll’ve": "I will have","I’m": "I am", "I’ve": "I have", "i’d": "i would",
                          "i’d’ve": "i would have", "i’ll": "i will",  "i’ll’ve": "i will have","i’m": "i am", "i’ve": "i have", "isn’t": "is not", "it’d": "it would",
                          "it’d’ve": "it would have", "it’ll": "it will", "it’ll’ve": "it will have","it’s": "it is", "let’s": "let us", "ma’am": "madam",
                          "mayn’t": "may not", "might’ve": "might have","mightn’t": "might not","mightn’t’ve": "might not have", "must’ve": "must have",
                          "mustn’t": "must not", "mustn’t’ve": "must not have", "needn’t": "need not", "needn’t’ve": "need not have","o’clock": "of the clock",
                          "oughtn’t": "ought not", "oughtn’t’ve": "ought not have", "shan’t": "shall not", "sha’n’t": "shall not", "shan’t’ve": "shall not have",
                          "she’d": "she would", "she’d’ve": "she would have", "she’ll": "she will", "she’ll’ve": "she will have", "she’s": "she is",
                          "should’ve": "should have", "shouldn’t": "should not", "shouldn’t’ve": "should not have", "so’ve": "so have","so’s": "so as",
                          "this’s": "this is","that’d": "that would", "that’d’ve": "that would have", "that’s": "that is", "there’d": "there would",
                          "there’d’ve": "there would have", "there’s": "there is", "here’s": "here is","they’d": "they would", "they’d’ve": "they would have",
                          "they’ll": "they will", "they’ll’ve": "they will have", "they’re": "they are", "they’ve": "they have", "to’ve": "to have",
                          "wasn’t": "was not", "we’d": "we would", "we’d’ve": "we would have", "we’ll": "we will", "we’ll’ve": "we will have", "we’re": "we are",
                          "we’ve": "we have", "weren’t": "were not", "what’ll": "what will", "what’ll’ve": "what will have", "what’re": "what are",
                          "what’s": "what is", "what’ve": "what have", "when’s": "when is", "when’ve": "when have", "where’d": "where did", "where’s": "where is",
                          "where’ve": "where have", "who’ll": "who will", "who’ll’ve": "who will have", "who’s": "who is", "who’ve": "who have",
                          "why’s": "why is", "why’ve": "why have", "will’ve": "will have", "won’t": "will not", "won’t’ve": "will not have",
                          "would’ve": "would have", "wouldn’t": "would not", "wouldn’t’ve": "would not have", "y’all": "you all",
                          "y’all’d": "you all would","y’all’d’ve": "you all would have","y’all’re": "you all are","y’all’ve": "you all have",
                          "you’d": "you would", "you’d’ve": "you would have", "you’ll": "you will", "you’ll’ve": "you will have",
                          "you’re": "you are", "you’ve": "you have","ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                          "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                          "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                          "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                          "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                          "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                          "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                          "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                          "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                          "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                          "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                          "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                          "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                          "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                          "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                          "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                          "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                          "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                          "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                          "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                          "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                          "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                          "you're": "you are", "you've": "you have","n't":'not'}

def cleaning(doc, contraction_mapping):
    clean = []
    for i in tqdm(doc):
        low = str(i).lower()
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
        clean.append(new)
    return clean

def process_data(data_type, contraction_mapping):
    print(f"Cleaning source {data_type} data")
    with open(f"{data_type}_document_cnn.txt", 'r') as file:
        doc = file.readlines()
    final_data = cleaning(doc, contraction_mapping)
    with open(f"src_{data_type}_cnn.txt", 'w') as file:
        for summary in tqdm(final_data):
            file.write(summary + '\n')

    print(f"Cleaning target {data_type} data")
    with open(f"{data_type}_title_cnn.txt", 'r') as file:
        doc = file.readlines()
    final_data = cleaning(doc, contraction_mapping)
    with open(f"tgt_{data_type}_cnn.txt", 'w') as file:
        for summary in tqdm(final_data):
            file.write(summary + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process training or testing data")
    parser.add_argument("--type", required=True, choices=["train", "test", "valid"], help="Specify whether to process train or test data")
    args = parser.parse_args()
    process_data(args.type, contraction_mapping)
    print("Done.")
