import re
import os
from tqdm import tqdm
from bs4 import BeautifulSoup

contraction_mapping = {
    "ain't": "is not",
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have"
}

def cleaning(doc,contraction_mapping):
    clean=[]
    for i in tqdm(doc):
        low=str(i).lower()
        soup=BeautifulSoup(low,'lxml')
        low=soup.text
        low=re.sub(" '","'",low)
        low=re.sub(" n't","n't",low)
        sent=[]
        for m in (low.split()):
            if m in contraction_mapping:
                sent.append(contraction_mapping[m])
            else:
                sent.append(m)
        jnt=' '.join(sent)
        low=re.sub("'s","",jnt)
        low=re.sub("’s","",low)
        new=re.sub("\n",'',low)
        new=re.sub(r'[\$\"\(\)\)\#\:\@\=\>\<\-\`\-\-\/\;\‘\£\%\*\—]',' ',new)
        new=re.sub(",",'',new)
        new=re.sub('\!','.',new)
        new=re.sub('\?','.',new)
        new=re.sub("'",'',new)
        new=re.sub("°",'',new)
        new=re.sub("\.\.\.",'.',new)
        new=re.sub(r"[^a-zA-Z0-9]",' ',new)
        new=(re.sub(r'[\s]+',' ',new)).strip()
        clean.append(new)
    return clean

# train data
print("Cleaning Source Training Data")
with open("train_document.txt",'r') as file:
    doc=file.readlines()
final_data_train=cleaning(doc,contraction_mapping)

with open("src_train.txt",'w') as file:
    for summary in tqdm(final_data_train):
        file.write(summary+'\n')

print("Cleaning Target Training Data")
with open("train_title.txt",'r') as file:
    doc=file.readlines()
final_data_train=cleaning(doc,contraction_mapping)

with open("tgt-train.txt",'w') as file:
    for summary in tqdm(final_data_train):
        file.write(summary+'\n')

# test data
print("Cleaning Source Test Data")
with open("test_document.txt",'r') as file:
    doc=file.readlines()
final_data_test=cleaning(doc,contraction_mapping)

with open("src_test.txt",'w') as file:
    for summary in tqdm(final_data_test):
        file.write(summary+'\n')

print("Cleaning Target Test Data")
with open("test_title.txt",'r') as file:
    doc=file.readlines()
final_data_test=cleaning(doc,contraction_mapping)

with open("tgt-test.txt",'w') as file:
    for summary in tqdm(final_data_test):
        file.write(summary+'\n')
