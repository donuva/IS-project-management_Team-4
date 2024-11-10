import re
import os
from tqdm import tqdm
from bs4 import BeautifulSoup
import argparse

def process_article(article_path):
    with open(article_path, 'r') as article:
        print("Article:", article_path)
        art = article.read()
        x = re.sub(r'\s', ' ', art)
        x = re.sub('\@body', '\n@body', x)
        x = re.sub('\@summary', '\n@summary', x)
        with open("data.txt", 'w') as f:
            f.write(x)
        with open("data.txt", 'r') as f:
            doc = f.readlines()
        summary = []
        for i in doc:
            i = re.sub(r'\s', ' ', i)
            if '@title' in i:
                title = i
            if '@body' in i:
                body = i

        if len(body.split()) > 25 and len(title.split()) > 3:
            with open('train_title.txt', 'a') as head, \
                 open('train_document.txt', 'a') as document, \
                 open('train_doc_name.txt', 'a') as doc_name:
                head.write(re.sub('\@title', '', title) + "\n")
                document.write(re.sub('\@body', '', body) + "\n")
                doc_name.write(os.path.basename(article_path) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from articles.")
    parser.add_argument("-p", "--article_path", required=True, help="Path to the article directory")
    args = parser.parse_args()

    base = args.article_path  #get the article path from the command line argument
    article_list = os.listdir(base)

    for article_name in article_list:
        article_path = os.path.join(base, article_name)
        process_article(article_path)

    print("Done.")