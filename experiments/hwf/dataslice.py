from argparse import ArgumentParser
import os
import json
import random

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--input", type=str, default="expr_train.json")
  parser.add_argument("--output", type=str, default="expr_train_0.5.json")
  parser.add_argument("--perc", type=float, default=0.5)
  parser.add_argument("--seed", type=int, default=1234)
  args = parser.parse_args()

  # Set random seed
  random.seed(args.seed)

  # Load input file
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data/HWF/"))
  input_file = json.load(open(os.path.join(data_root, args.input)))

  # Shuffle the file and pick only the top arg.perc
  random.shuffle(input_file)
  end_index = int(len(input_file) * args.perc)
  input_file = input_file[0:end_index]

  # Output the file
  json.dump(input_file, open(os.path.join(data_root, args.output), "w"))
