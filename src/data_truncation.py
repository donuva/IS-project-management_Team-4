import argparse
import re
from tqdm import tqdm

def data_truncation(type):
    with open(f"src_{type}.txt",'r') as summary:
        doc=summary.readlines()
        summary= [re.sub('\n','',i) for i in doc]
        
    with open(f"./truncated_docs/t_src_{type}.txt",'w') as file:  
        final=[]
        for line in tqdm(summary):
            split=line.split()
            if len(split) > 110:
                trunc_list=split[0:110]
                final.append(' '.join(trunc_list))
            else:
                final.append(line)
        for line in final:
            file.write(line+"\n")
            
    # Target data truncation
    with open(f"tgt_{type}.txt",'r') as summary:
        doc=summary.readlines()
        summary= [re.sub('\n','',i) for i in doc]
        
    with open(f"./truncated_docs/t_tgt_{type}.txt",'w') as file:  
        final=[]
        for line in tqdm(summary):
            split=line.split()
            if len(split) > 26:
                trunc_list=split[0:26]
                final.append(' '.join(trunc_list))
            else:
                final.append(line)
        for line in final:
            file.write(line+"\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from articles.")
    parser.add_argument("--type", required=True, choices=["train", "test", "valid"], help="Specify if this is for training, testing or validating.")
    args = parser.parse_args()
    print("Processing...")
    data_truncation(args.type)
    print("Done.")