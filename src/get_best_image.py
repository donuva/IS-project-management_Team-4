import argparse
import matplotlib.pyplot as plt
import re

def process_bleu_scores(type):
    bleu_scores_file = f"bleu_scores_captions_{type}.txt"
    best_caption_file = f"best_caption_{type}.txt"
    
    with open(bleu_scores_file, "r") as f:
        arBleu_list = f.readlines()

    with open(best_caption_file, "w") as f:
        max_bleu = float(arBleu_list[0].split()[1][5:])
        pre_ar_name = arBleu_list[0].split()[0].split("_")[0]
        for index in range(1, len(arBleu_list)):
            article = arBleu_list[index]
            ar_name_i = article.split()[0].split("_")[0]
            id_i = article.split()[0].split("_")[1]
            bleu_score_i = float(article.split()[1][5:])

            if ar_name_i == pre_ar_name and index > 0:
                if bleu_score_i > max_bleu:
                    max_bleu = bleu_score_i
                    best_index = id_i

                print(pre_ar_name, " ", max_bleu)
            else:
                print(pre_ar_name)
                f.write(f"{pre_ar_name}_{best_index} \n")
                pre_ar_name = ar_name_i
                max_bleu = -1e9999

    with open(best_caption_file, "r") as f:
        best_list = f.readlines()

    # Original target summary
    target_file = f"tgt-{type}.txt"
    with open(target_file, 'r') as file:
        original = file.readlines()
        original = [re.sub("\n", '', i) for i in original]

    # Best Image index file
    with open(best_caption_file, 'r') as file:
        best_cap = file.readlines()
        best_cap = [re.sub(" \n", '', cap) for cap in best_cap]

    for i in range(4):  
        print(i + 1)
        print("Original Summary : ", original[i])
        print("Image Caption : ", best_cap[i])

        # Uncomment these lines to display images if required
        # print("Generated Image")
        # x = plt.imread(f"./img/{best_cap[i]}.jpg")
        # plt.imshow(x / 255.)
        # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process BLEU scores and find the best captions.")
    parser.add_argument("--type", required=True, choices=["train", "test"], help="Specify if this is for training or testing.")
    args = parser.parse_args()

    process_bleu_scores(args.type)
