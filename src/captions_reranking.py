import re
import os
from bs4 import BeautifulSoup
import argparse
from rerankers import Reranker
from tqdm import tqdm

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

def rerank(model, target_file, article_names_file, caption_dir, img_dir, type):
    target_summaries = []
    doc_name = []
    with open(article_names_file, 'r') as f:
        for line in f:
            doc_name.append(line.strip())
    print("Total articles found: ", len(doc_name))

    with open(target_file, 'r') as f:
        for line in f:
            target_summaries.append(line.strip())
    print("Total summaries found: ", len(target_summaries))

    # lists to store
    all_image_names = []
    all_caption_names = []
    all_captions = []
    for i, article_name in tqdm(enumerate(doc_name), total=len(doc_name), desc="Processing Articles"):
        article_id = re.sub(r'\.txt$', '', article_name)

        images = get_images(article_id, img_dir)
        images = sorted(images, key=sort_key)

        captions = get_captions(article_id, caption_dir)
        captions = sorted(captions, key=sort_key)

        # Find and remove extra element from either list
        images, captions = find_and_remove_diff(images, captions)

        all_image_names.append(images)
        all_caption_names.append(captions)

        caption_list = []
        for caption_name in captions:
            caption_path = os.path.join(caption_dir, caption_name)
            with open(caption_path, 'r') as caption_file:
                caption_content = caption_file.read().strip()
                caption_list.append(caption_content)
            
        all_captions.append(caption_list)

    if model == 'colbert':
        ranker = Reranker('colbert',
            model_type="colbert",
            verbose = 1, # How verbose the reranker will be. Defaults to 1, setting it to 0 will suppress most messages.
            dtype = None, # Which dtype the model should use. If None will figure out if your platform + model combo supports fp16 and use it if so, other fp32.
            device = None,  # Which device the model should use. If None will figure out what the most powerful supported platform available is (cuda > mps > cpu)
            batch_size = 16,  # The batch size the model will use. Defaults to 16
            query_token = "[unused0]",  # A ColBERT-specific argument. The token that your model prepends to queries.
            document_token = "[unused1]",  # A ColBERT-specific argument. The token that your model prepends to documents.
        )
    elif model == 'cross-encoder/ms-marco-MiniLM-L-6-v2':
        ranker = Reranker('cross-encoder/ms-marco-MiniLM-L-6-v2',
            model_type="cross-encoder",
            verbose = 1, # How verbose the reranker will be. Defaults to 1, setting it to 0 will suppress most messages.
            dtype = None, # Which dtype the model should use. If None will figure out if your platform + model combo supports fp16 and use it if so, other fp32.
            device = None,  # Which device the model should use. If None will figure out what the most powerful supported platform available is (cuda > mps > cpu)
            batch_size = 16,  # The batch size the model will use. Defaults to 16
        )
    elif model == 't5':
        ranker = Reranker('t5',
            model_type="t5",
            verbose = 1, # How verbose the reranker will be. Defaults to 1, setting it to 0 will suppress most messages.
            dtype = None, # Which dtype the model should use. If None will figure out if your platform + model combo supports fp16 and use it if so, other fp32.
            device = None,  # Which device the model should use. If None will figure out what the most powerful supported platform available is (cuda > mps > cpu)
            batch_size = 16,  # The batch size the model will use. Defaults to 16
            token_false = "auto", # The output token corresponding to non-relevance.
            token_true = "auto", # The output token corresponding to relevance.
            return_logits = False, # Whether to return a normalised score or the raw logit for `token_true`.
        )

    output_file = f"{type}_top_ranked_captions.txt"
    with open(output_file, 'w') as outfile:
        for i, query in tqdm(enumerate(target_summaries), total=len(target_summaries), desc="Ranking Queries"):
            print(f"Query {i}: {query}")
            print(f"Total docs: {len(all_captions[i])}")
            if all_captions[i]: 
                docs = all_captions[i]
                results = ranker.rank(query=query, docs=docs)
                top_result = results.top_k(1)[0]

                document_id = top_result.document.doc_id
                best_caption_file = all_caption_names[i][document_id].strip()

                print(f"Best caption from reranked model: {docs[document_id]}")
                print(f"Best caption file: {best_caption_file}")

                caption_path = caption_dir + best_caption_file
                with open(caption_path, 'r') as caption_file:
                    caption_content = caption_file.read()
                    print(caption_content)

                outfile.write(best_caption_file + '\n')
            else:
                print("Warning: No captions found for this query")
            print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerank captions.")
    parser.add_argument("-t", "--target_file", required=True, help="Path to the target file (Ex: test_title.txt/pred_mpt.txt).")
    parser.add_argument("-a", "--article_names_file", required=True, help="Path to the file containing all article names (Ex: test_doc_name.txt).") 
    parser.add_argument("-c", "--caption_dir", required=True, help="Path to the caption dir.") 
    parser.add_argument("-i", "--img_dir", required=True, help="Path to the image dir.") 
    parser.add_argument("-m", "--rerank_model", choices=['colbert', 't5', 'cross-encoder/ms-marco-MiniLM-L-6-v2'], default='colbert', help="Rerank model.") 
    parser.add_argument("--type", required=True, choices=["train", "test", "valid"], help="Specify if this is for training, testing or validating.")
    args = parser.parse_args()

    print("Processing...")
    rerank(args.rerank_model, args.target_file, args.article_names_file,
           args.caption_dir, args.img_dir, args.type)
    print("Done.")
  
