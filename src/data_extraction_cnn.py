import re
import os
from tqdm import tqdm
from bs4 import BeautifulSoup
import argparse
import pandas as pd

def process_article(df, type):
    with open(f'{type}_title_cnn.txt', 'a') as head, \
         open(f'{type}_document_cnn.txt', 'a') as document:
        for index, row in df.iterrows():
            highlights = row['highlights'].replace('\n', ' ')
            head.write(f"{highlights}\n")
            document.write(f"{row['article']}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from articles.")
    parser.add_argument("-p", "--data_file", required=True, help="Path to the csv file")
    parser.add_argument("-n", "--num_articles", type=int, default=-1, help="Number of articles to process (-1 for all)") 
    parser.add_argument("--type", required=True, choices=["train", "test", "valid"], help="Specify if this is for training or testing")
    args = parser.parse_args()

    article_df = pd.read_csv(args.data_file)

    if args.num_articles != -1: 
        article_df = article_df[:args.num_articles] 

    print("Processing...")
    process_article(article_df, args.type)

    print("Done.")