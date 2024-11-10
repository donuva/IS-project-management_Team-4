import re
import os
import argparse

def write_images_to_file(image_folder_path, doc_file):
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
        for image_i in os.listdir(image_folder_path):  # Get images from folder
            if article_i in image_i:
                article_i_image.append(re.sub('\.jpg', '', image_i))
        if article_i_image == []:
            image_list.append('None')
        else:
            image_list.append(article_i_image)

    with open("Test_Image_names.txt", "w") as file:
        for i in image_list:
            file.write('\t'.join(i) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write image file names to file.")
    parser.add_argument("-i", "--doc_file", required=True, help="Name of the input file") 
    parser.add_argument("-p", "--image_folder_path", required=True, help="Path to the image folder")
    args = parser.parse_args()

    write_images_to_file(args.image_folder_path, args.doc_file)
    print("Done.")