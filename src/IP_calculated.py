import argparse

def cal(args):
  #
  top_ranked = []
  with open(args.predict, "r") as f:
    top_ranked = f.readlines()

  predict=[]
  for line in top_ranked:
    line=line.replace("_concat.caption\n", ".jpg")
    #print(line)
    predict.append(line)
  #
  with open(args.annotate, "r") as f:
    tgt = f.readlines()

  truth = []
  for line in tgt:
    #print(line)
    if "None" not in line:
      truth.extend(line.split())
  #
  num_true = 0
  for line in predict:
    #print(line)
    if line in truth:
      num_true += 1

  return round(num_true/len(predict), 2)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--predict', required=True)
  parser.add_argument('--annotate', required=True)
  args = parser.parse_args()
  print(cal(args))
import argparse

def cal(args):
  #
  top_ranked = []
  with open(args.predict, "r") as f:
    top_ranked = f.readlines()

  predict=[]
  for line in top_ranked:
    line=line.replace("_concat.caption\n", ".jpg")
    #print(line)
    predict.append(line)
  #
  with open(args.annotate, "r") as f:
    tgt = f.readlines()

  truth = []
  for line in tgt:
    #print(line)
    if "None" not in line:
      truth.extend(line.split())
  #
  num_true = 0
  for line in predict:
    #print(line)
    if line in truth:
      num_true += 1

  return round(num_true/len(predict), 2)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--predict', required=True)
  parser.add_argument('--annotate', required=True)
  args = parser.parse_args()
  cal(args)

