from argparse import ArgumentParser
import os
import json

if __name__ == "__main__":
  parser = ArgumentParser("hwf/datastats")
  parser.add_argument("--dataset", type=str, default="expr_train.json")
  args = parser.parse_args()

  # Get the dataset
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data/HWF"))
  data = json.load(open(os.path.join(data_root, args.dataset)))

  # Compute stats
  lengths = {}
  for datapoint in data:
    if len(datapoint["img_paths"]) in lengths: lengths[len(datapoint["img_paths"])] += 1
    else: lengths[len(datapoint["img_paths"])] = 1
  print(lengths)
