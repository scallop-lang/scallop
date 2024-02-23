import os
import json
import random
from argparse import ArgumentParser
from tqdm import tqdm
import math

import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from PIL import Image

import scallopy
import math

from run_with_hwf_parser import HWFNet, SymbolNet

class HWFDataset(torch.utils.data.Dataset):
  def __init__(self, root: str, prefix: str, split: str):
    super(HWFDataset, self).__init__()
    self.root = root
    self.split = split
    self.metadata = json.load(open(os.path.join(root, f"HWF/{prefix}_{split}.json")))
    self.img_transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.5,), (1,))
    ])

  def __getitem__(self, index):
    sample = self.metadata[index]

    # Input is a sequence of images
    img_seq = []
    for img_path in sample["img_paths"]:
      img_full_path = os.path.join(self.root, "HWF/Handwritten_Math_Symbols", img_path)
      img = Image.open(img_full_path).convert("L")
      img = self.img_transform(img)
      img_seq.append(img)
    img_seq_len = len(img_seq)

    # Output is the "res" in the sample of metadata
    res = sample["res"]

    # Output the expression
    expr = sample["expr"]

    # Return (input, output) pair
    return (img_seq, img_seq_len, expr, res)

  def __len__(self):
    return len(self.metadata)

  @staticmethod
  def collate_fn(batch):
    max_len = max([img_seq_len for (_, img_seq_len, _, _) in batch])
    zero_img = torch.zeros_like(batch[0][0][0])
    pad_zero = lambda img_seq: img_seq + [zero_img] * (max_len - len(img_seq))
    img_seqs = torch.stack([torch.stack(pad_zero(img_seq)) for (img_seq, _, _, _) in batch])
    img_seq_len = torch.stack([torch.tensor(img_seq_len).long() for (_, img_seq_len, _, _) in batch])
    results = torch.stack([torch.tensor(res) for (_, _, _, res) in batch])
    exprs = [expr for (_, _, expr, _) in batch]
    return (img_seqs, img_seq_len, exprs, results)


def hwf_loader(data_dir, batch_size, prefix):
  return torch.utils.data.DataLoader(
    HWFDataset(data_dir, prefix, "test"),
    collate_fn=HWFDataset.collate_fn,
    batch_size=batch_size,
    shuffle=False,
  )

def eval_result_eq(a, b, threshold=0.01):
  return abs(a - b) < threshold

if __name__ == "__main__":
  # Command line arguments
  parser = ArgumentParser("test_hwf_model")
  parser.add_argument("--model-name", type=str, default="hwf.pkl")
  parser.add_argument("--dataset-prefix", type=str, default="expr")
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--seed", type=int, default=12345)
  parser.add_argument("--cuda", action="store_true")
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument("--verbose", action="store_true")
  args = parser.parse_args()

  # Parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)
  if args.cuda:
    if torch.cuda.is_available(): device = torch.device(f"cuda:{args.gpu}")
    else: raise Exception("No cuda available")
  else: device = torch.device("cpu")

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  test_loader = hwf_loader(data_dir, batch_size=args.batch_size, prefix=args.dataset_prefix)

  # Model
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/hwf"))
  hwf_net = torch.load(open(os.path.join(model_dir, args.model_name), "rb"))

  # Testing
  hwf_net.eval()
  with torch.no_grad():
    processed_count, correct_count = 0, 0
    dataset_iter = test_loader if args.verbose else tqdm(test_loader)
    for img_seq, img_seq_len, exprs, labels in dataset_iter:
      batch_size, formula_length, _, _, _ = img_seq.shape

      # Predict per character symbol
      symbol_distr = hwf_net.symbol_cnn(img_seq.flatten(start_dim=0, end_dim=1)).view(batch_size, formula_length, -1)
      exprs_pred = ["".join([hwf_net.symbols[torch.argmax(s).item()] for s in task_symbols_distrs]) for task_symbols_distrs in symbol_distr]

      # Do the prediction
      (output_mapping, y_pred) = hwf_net(img_seq, img_seq_len)
      y_pred = y_pred.to("cpu")

      # Get the predictions
      y_pred_index = torch.argmax(y_pred, dim=1)

      # Iterate through all examples
      batch_correct_count = 0
      for i in range(batch_size):
        expr = exprs[i]
        expr_pred = exprs_pred[i]
        y = labels[i]
        y_pred = output_mapping[y_pred_index[i]]
        is_correct = eval_result_eq(y, y_pred)
        if is_correct:
          batch_correct_count += 1
        correct_str = "[Correct]" if is_correct else "[Incorrect]"
        if args.verbose:
          print(f"{correct_str} Ground Truth Expr: {expr}, Predicted Expr: {expr_pred}, Ground Truth: {y}, Computed: {y_pred}")

      # Update the progress bar
      processed_count += batch_size
      correct_count += batch_correct_count
      if not args.verbose:
        perc = float(correct_count) / float(processed_count) * 100.0
        dataset_iter.set_description(f"Correct: {correct_count}/{processed_count} ({perc:.4f}%)")

    # If verbose, print the accuracy at the end
    if args.verbose:
      perc = float(correct_count) / float(processed_count) * 100.0
      print(f"Overall Correctness: {correct_count}/{processed_count} ({perc:.4f}%)")
