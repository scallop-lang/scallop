import os
import json

class GSM8KDataset:
  def __init__(self):
    self.dataset_loc = "../../../gsm8k/test.jsonl"
    
    with open(self.dataset_loc) as dataset:
      self.data = list(json.loads(obj) for obj in dataset)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    question = self.data[i]["question"]
    answer = float(self.data[i]["answer"].split()[-1].replace(',', '')) # The answer is always given at the end as the last word

    return (question, answer)
