import os
import csv
import json

class ModifiedCLUTRRDataset:
    def __init__(self, root=".", dataset="data_089907f8", split="test", difficulty=range(2,11)):
        self.dataset_dir = os.path.join(root, f"CLUTRR/{dataset}/")
        self.file_names = [os.path.join(self.dataset_dir, f"1.{d}_{split}.csv") for d in difficulty]
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

with open("data_cot.json") as file:
    data = json.load(file)

datasets = {d: set(ctx + " Who is " + query[1] + " to " + query[0] + "?" for (ctx, query), _ in ModifiedCLUTRRDataset(difficulty=[d])) for d in range(2, 11)}

# index 1 something to do with matched to multiple, index 0 will hold any that did not match any level
accuracy_mapping = [[0, 0] for _ in range(11)]

for ex in data["data"]:
    matched = False
    for d in range(2, 11):
        if ex["question"] in datasets[d]:
            if matched:
                accuracy_mapping[1][ex["score"]] += 1
            accuracy_mapping[d][ex["score"]] += 1
            matched = True
    if not matched:
        accuracy_mapping[0][ex["score"]] += 1

assert(accuracy_mapping[0] == [0, 0])
assert(accuracy_mapping[1] == [0, 0])

for d in range(2, 11):
    correct = accuracy_mapping[d][1]
    total = len(ModifiedCLUTRRDataset(difficulty = [d]))
    assert(correct + accuracy_mapping[d][0] == total)
    print(f"Level {d}: {correct} / {total} = {correct / total * 100}%")