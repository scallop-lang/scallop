from argparse import ArgumentParser
import os
import json

if __name__ == "__main__":
  parser = ArgumentParser("hwf/datamerge")
  parser.add_argument("inputs", action="store", nargs="*")
  parser.add_argument("--output", type=str, default="expr_merged.json")
  args = parser.parse_args()

  # Get the list of files
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data/HWF"))
  all_data = []
  for file in args.inputs:
    print(f"Loading file {file}")
    data = json.load(open(os.path.join(data_root, file)))
    all_data += data

  # Dump the result
  json.dump(all_data, open(os.path.join(data_root, args.output), "w"))
