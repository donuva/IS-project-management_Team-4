import re
import os
import argparse
from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

def sort_key(filename):
  """Extracts the ID after the underscore and converts it to an integer."""
  match = re.search(r'_(\d+)\.(jpg|caption|txt)$', filename)
  return int(match.group(1)) if match else -1

def get_captions(article_id, caption_dir):
  caption_paths = []
  for filename in os.listdir(caption_dir):
    if filename.startswith(article_id) and filename.endswith(".caption"):
      caption_paths.append(os.path.join(caption_dir, filename).strip().split('/')[-1])
  return caption_paths

def get_images(article_id, img_dir):
  img_paths = []
  for filename in os.listdir(img_dir):
    if filename.startswith(article_id) and filename.endswith(".jpg"):
      img_paths.append(os.path.join(img_dir, filename).strip().split('/')[-1])
  return img_paths

def find_and_remove_diff(list1, list2):
    """Finds and removes the extra element from either list1 or list2 to make them correspond."""

    # Extract IDs from filenames
    ids1 = [sort_key(filename) for filename in list1]
    ids2 = [sort_key(filename) for filename in list2]

    # Find the difference in IDs
    diff = list(set(ids1) ^ set(ids2))

    # Remove elements with the extra ID from the appropriate list
    if diff:
        extra_id = diff[0]
        if extra_id in ids1:
            list1 = [filename for filename in list1 if sort_key(filename) != extra_id]
        else:
            list2 = [filename for filename in list2 if sort_key(filename) != extra_id]

    return list1, list2

def main(model_name, batch_size, all_article_names, caption_dir, img_dir, concat_caption_dir):
    if model_name != 'vikhyatk/moondream2':
        print("Model not supported.")
        return

    all_articles = []
    with open(all_article_names, 'r') as f:
      for line in f:
        all_articles.append(line.strip())
    print("Total articles found: ", len(all_articles))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    revision = "2024-08-26"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, revision=revision).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

    preprocess = transforms.Compose([
        transforms.Resize((378, 378)), #change for different model
        transforms.ToTensor(),
    ])

    concat_captions = []
    all_caption_names = [] 
    for article_name in tqdm(all_articles, desc="Processing Articles"):
        article_id = re.sub(r'\.txt$', '', article_name)

        images = sorted(get_images(article_id, img_dir), key=sort_key)
        captions = sorted(get_captions(article_id, caption_dir), key=sort_key)
        images, captions = find_and_remove_diff(images, captions)

        all_caption_names.append(captions)

        # Process images and captions in batches
        for i in range(0, len(images), batch_size):
            batch_images = []
            valid_captions = []

            for img_path, caption_path in zip(images[i : i + batch_size], captions[i : i + batch_size]):
                try:
                    img = Image.open(os.path.join(img_dir, img_path)).convert("RGB")
                    img_tensor = preprocess(img)  
                    if  img_tensor.shape == torch.Size([3, 378, 378]):  
                        batch_images.append(img_tensor)
                        caption_path = os.path.join(caption_dir, caption_path)
                        with open(caption_path, 'r') as caption_file:
                            caption = caption_file.read().strip()
                        valid_captions.append(caption)
                    else:
                        print(f"Skipping image {img_path} due to size mismatch.")
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")

            if not batch_images:
                continue

            # Stack images to create a batch (B, C, H, W)
            batch_images = torch.stack(batch_images).to(device)

            with torch.no_grad():
                enc_images = model.encode_image(batch_images)  
                batch_captions = []
                for enc_image, caption in zip(enc_images, valid_captions):
                    detailed_description = model.answer_question(
                        enc_image.unsqueeze(0), 
                        "Describe this image with as much details as possible.",
                        tokenizer
                    ).strip()
                    batch_captions.append(caption.strip() + ". " + detailed_description)

            concat_captions.append(batch_captions)

    # write 
    for captions, caption_names in tqdm(zip(concat_captions, all_caption_names), total=len(concat_captions), desc="Writing Captions"):
        for caption, caption_name in zip(captions, caption_names):
            print(caption_name)
            new_name = re.sub(r'\.caption$', r'_concat.caption', caption_name)
            path = os.path.join(concat_caption_dir, new_name)
            print(caption)
            try:
                with open(path, 'w') as f:
                    f.write(caption)
            except Exception as e:
                print(f"Error writing caption {caption_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerank captions.")
    parser.add_argument("-a", "--article_names_file", required=True, help="Path to the file containing all article names (Ex: test_doc_name.txt).") 
    parser.add_argument("-c", "--caption_dir", required=True, help="Path to the caption dir.") 
    parser.add_argument("-cc", "--concat_caption_dir", default='./concat_caption', help="Path to the concat caption dir.") 
    parser.add_argument("-i", "--img_dir", required=True, help="Path to the image dir.") 
    parser.add_argument("-m", "--caption_model", default='vikhyatk/moondream2', help="Caption model.") 
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size.") 
    args = parser.parse_args()

    print("Processing...")
    main(args.caption_model, args.batch_size, args.article_names_file, args.caption_dir, args.img_dir, args.concat_caption_dir)
    print("Done.")
