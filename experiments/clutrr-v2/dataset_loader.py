import os
import csv

class CLUTRRDataset:
  def __init__(self, root=".", dataset="data_089907f8", split="test"):
    self.dataset_dir = os.path.join(root, f"CLUTRR/{dataset}/")
    self.file_names = [os.path.join(self.dataset_dir, d) for d in os.listdir(self.dataset_dir) if f"_{split}.csv" in d]
    self.data = [row for f in self.file_names for row in list(csv.reader(open(f)))[1:]]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    # Context is a list of sentences
    context = self.data[i][2].strip()

    # Remove square brackets
    context = context.replace('[', '')
    context = context.replace(']', '')

    # Query is of type (sub, obj)
    query_sub_obj = eval(self.data[i][3])
    query = (query_sub_obj[0], query_sub_obj[1])

    # Answer is one of 20 classes such as daughter, mother, ...
    answer = self.data[i][5]
    return ((context, query), answer)