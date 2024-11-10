import re
import os
import argparse
import nltk
from nltk.translate.bleu_score import sentence_bleu
nltk.download('punkt')

def get_image_list(image_folder_path, doc_file):
  with open(doc_file, 'r') as doc:
      article_list = doc.readlines()

  clean_ar_list = []
  for i in article_list:
      d1 = re.sub('\n', "", i)
      d2 = re.sub('\.txt', "", d1)
      clean_ar_list.append(d2)

  image_list = []
  for article_i in clean_ar_list:
      article_i_image = []
      for image_i in os.listdir(image_folder_path):  # get images from folder
          if article_i in image_i:
              article_i_image.append(re.sub('\.jpg', '', image_i))
      if article_i_image == []:
          image_list.append('None')
      else:
          image_list.append(article_i_image)

  return image_list

def read_captions_from_list(caption_list):
  all_captions = []
  all_captions_file_name = []
  for captions_image in caption_list:
    for file_path in captions_image:
        if file_path == "None":  #skip images with no captions
            continue
        with open(caption_dir + file_path, "r") as f:
            captions = f.readlines()
            for cap in captions:
                file_name_without_ext = os.path.splitext(file_path)[0]
                all_captions_file_name.append(file_name_without_ext)
                all_captions.append(cap.strip())
  return all_captions_file_name, all_captions

def calculate_bleu_from_file(file_path, all_captions):
  bleu_scores = []

  with open(file_path, 'r') as file:
    for line in file:
      target_sentence = line.strip()
      target_tokens = nltk.word_tokenize(target_sentence)

      for captions in caption_list:
        save = 0
        article_name = captions[0]
        article_name = re.sub(r'_\d+\.caption', '',article_name)
        for i in range(len(captions)):
          if captions[i] == 'None':
            continue
          reference_caption = all_captions[save + i]
          reference_tokens = nltk.word_tokenize(reference_caption)
          score = sentence_bleu([reference_tokens], target_tokens)
          # print(f"Article Name: {article_name}")
          # print(f"Target: {target_sentence}")
          # print(f"Reference: {reference_caption}")
          # print(f"BLEU Score: {score}")
          # print("\n")
          bleu_scores.append(score)

        save += len(captions)

  return bleu_scores

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Write images for articles to file.")
  parser.add_argument("-d", "--doc_file", required=True, help="Name of file containings all documents") 
  parser.add_argument("-i", "--image_folder_path", required=True, help="Path to the image folder")
  parser.add_argument("-c", "--caption_folder_path", required=True, help="Path to the caption folder")
  parser.add_argument("-t", "--target_file", required=True, help="Target file to calculate bleu score with")
  args = parser.parse_args()

  image_list = get_image_list(args.image_folder_path, args.doc_file)
  sorted_image_list = []
  for sublist in image_list:
    sorted_image_list.append(sorted(sublist, key=lambda x: int(re.search(r'_(\d+)$', x).group(1))))
  
  caption_dir = args.caption_folder_path
  caption_list = []
  for image_names in sorted_image_list:
    if image_names == 'None':
      caption_list.append('None')
    temp = []
    for image_name in image_names:
      caption_path = f"{image_name}.caption"
      temp.append(caption_path)
    caption_list.append(temp)
  
  all_captions_file_name, all_captions = read_captions_from_list(caption_list)
  # write all captions to file
  with open("all_captions.txt", "w") as output_file:
    for file_name, caption in zip(all_captions_file_name, all_captions):
      output_file.write(file_name + " " + caption + "\n")

  # calculate bleu
  file_path = args.target_file
  bleu_scores = calculate_bleu_from_file(file_path, all_captions)

  # write bleu scores to file
  end_file_name = "end_file_name"
  with open("bleu_scores_captions.txt", "w") as f:
    for file_name, score in zip(all_captions_file_name, bleu_scores):
      f.write(f"{file_name} BLEU={score}\n")
    f.write(f"{end_file_name} BLEU={0.0}\n")
  print("BLEU scores saved to bleu_scores_captions.txt")
  print("Done.")