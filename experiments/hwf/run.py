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

class HWFDataset(torch.utils.data.Dataset):
  def __init__(self, root: str, prefix: str, split: str):
    super(HWFDataset, self).__init__()
    self.root = root
    self.split = split
    self.metadata = json.load(open(os.path.join(root, f"HWF/{prefix}_{split}.json")))
    self.img_transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.5,), (1,))])

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

    # Return (input, output) pair
    return (img_seq, img_seq_len, res)

  def __len__(self):
    return len(self.metadata)

  @staticmethod
  def collate_fn(batch):
    max_len = max([img_seq_len for (_, img_seq_len, _) in batch])
    zero_img = torch.zeros_like(batch[0][0][0])
    pad_zero = lambda img_seq: img_seq + [zero_img] * (max_len - len(img_seq))
    img_seqs = torch.stack([torch.stack(pad_zero(img_seq)) for (img_seq, _, _) in batch])
    img_seq_len = torch.stack([torch.tensor(img_seq_len).long() for (_, img_seq_len, _) in batch])
    results = torch.stack([torch.tensor(res) for (_, _, res) in batch])
    return (img_seqs, img_seq_len, results)


def hwf_loader(data_dir, batch_size, prefix):
  train_loader = torch.utils.data.DataLoader(HWFDataset(data_dir, prefix, "train"), collate_fn=HWFDataset.collate_fn, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(HWFDataset(data_dir, prefix, "test"), collate_fn=HWFDataset.collate_fn, batch_size=batch_size, shuffle=True)
  return (train_loader, test_loader)


class SymbolNet(nn.Module):
  def __init__(self):
    super(SymbolNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, stride = 1, padding = 1)
    self.conv2 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
    self.fc1 = nn.Linear(30976, 128)
    self.fc2 = nn.Linear(128, 14)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.max_pool2d(x, 2)
    x = F.dropout(x, p=0.25, training=self.training)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc2(x)
    return F.softmax(x, dim=1)


class HWFNet(nn.Module):
  def __init__(self, no_sample_k, sample_k, provenance, k):
    super(HWFNet, self).__init__()
    self.no_sample_k = no_sample_k
    self.sample_k = sample_k
    self.provenance = provenance
    self.symbols = [str(i) for i in range(10)] + ["+", "-", "*", "/"]

    # Symbol embedding
    self.symbol_cnn = SymbolNet()

    # Scallop context
    self.eval_formula = scallopy.Module(
      file=os.path.abspath(os.path.join(os.path.abspath(__file__), f"../scl/hwf_eval.scl")),
      non_probabilistic=["length"],
      input_mappings={"symbol": scallopy.Map({0: range(7), 1: self.symbols}, retain_k=sample_k, sample_dim=1, sample_strategy="categorical")},
      output_relation="result",
    )

  def forward(self, img_seq, img_seq_len):
    batch_size, formula_length, _, _, _ = img_seq.shape
    length = [[(l.item(),)] for l in img_seq_len]
    symbol = self.symbol_cnn(img_seq.flatten(start_dim=0, end_dim=1)).view(batch_size, formula_length, -1)
    (mapping, probs) = self.eval_formula(symbol=symbol, length=length)
    return ([v for (v,) in mapping], probs)


class Trainer():
  def __init__(self, train_loader, test_loader, device, model_root, model_name, learning_rate, no_sample_k, sample_k, provenance, k):
    self.network = HWFNet(no_sample_k, sample_k, provenance, k).to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.device = device
    self.loss_fn = F.binary_cross_entropy
    self.model_root = model_root
    self.model_name = model_name
    self.min_test_loss = 100000000.0

  def eval_result_eq(self, a, b, threshold=0.01):
    result = abs(a - b) < threshold
    return result

  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    train_loss = 0
    total_correct = 0
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    for (i, (img_seq, img_seq_len, label)) in enumerate(iter):
      (output_mapping, y_pred) = self.network(img_seq.to(device), img_seq_len.to(device))
      y_pred = y_pred.to("cpu")

      # Normalize label format
      batch_size, num_outputs = y_pred.shape
      y = torch.tensor([1.0 if self.eval_result_eq(l.item(), m) else 0.0 for l in label for m in output_mapping]).view(batch_size, -1)

      # Compute loss
      loss = self.loss_fn(y_pred, y)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      if not math.isnan(loss.item()):
        train_loss += loss.item()

      # Collect index and compute accuracy
      if num_outputs > 0:
        y_index = torch.argmax(y, dim=1)
        y_pred_index = torch.argmax(y_pred, dim=1)
        correct_count = torch.sum(torch.where(torch.sum(y, dim=1) > 0, y_index == y_pred_index, torch.zeros(batch_size).bool())).item()
      else:
        correct_count = 0

      # Stats
      num_items += batch_size
      total_correct += correct_count
      perc = 100. * total_correct / num_items
      avg_loss = train_loss / (i + 1)

      # Prints
      iter.set_description(f"[Train Epoch {epoch}] Avg loss: {avg_loss:.4f}, Accuracy: {total_correct}/{num_items} ({perc:.2f}%)")

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = 0
    test_loss = 0
    total_correct = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for i, (img_seq, img_seq_len, label) in enumerate(iter):
        (output_mapping, y_pred) = self.network(img_seq.to(device), img_seq_len.to(device))
        y_pred = y_pred.to("cpu")

        # Normalize label format
        batch_size, num_outputs = y_pred.shape
        y = torch.tensor([1.0 if self.eval_result_eq(l.item(), m) else 0.0 for l in label for m in output_mapping]).view(batch_size, -1)

        # Compute loss
        loss = self.loss_fn(y_pred, y)
        if not math.isnan(loss.item()):
          test_loss += loss.item()

        # Collect index and compute accuracy
        if num_outputs > 0:
          y_index = torch.argmax(y, dim=1)
          y_pred_index = torch.argmax(y_pred, dim=1)
          correct_count = torch.sum(torch.where(torch.sum(y, dim=1) > 0, y_index == y_pred_index, torch.zeros(batch_size).bool())).item()
        else:
          correct_count = 0

        # Stats
        num_items += batch_size
        total_correct += correct_count
        perc = 100. * total_correct / num_items
        avg_loss = test_loss / (i + 1)

        # Prints
        iter.set_description(f"[Test Epoch {epoch}] Avg loss: {avg_loss:.4f}, Accuracy: {total_correct}/{num_items} ({perc:.2f}%)")

    # Save model
    if test_loss < self.min_test_loss:
      self.min_test_loss = test_loss
      torch.save(self.network, os.path.join(self.model_root, self.model_name))

  def train(self, n_epochs):
    # self.test_epoch(0)
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)


if __name__ == "__main__":
  # Command line arguments
  parser = ArgumentParser("hwf")
  parser.add_argument("--model-name", type=str, default="hwf.pkl")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--no-sample-k", action="store_true")
  parser.add_argument("--sample-k", type=int, default=3)
  parser.add_argument("--dataset-prefix", type=str, default="expr")
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--do-not-use-hash", action="store_true")
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--top-k", type=int, default=3)
  parser.add_argument("--cuda", action="store_true")
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument("--jit", action="store_true")
  parser.add_argument("--recompile", action="store_true")
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
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/hwf"))
  if not os.path.exists(model_dir): os.makedirs(model_dir)
  train_loader, test_loader = hwf_loader(data_dir, batch_size=args.batch_size, prefix=args.dataset_prefix)

  # Training
  trainer = Trainer(train_loader, test_loader, device, model_dir, args.model_name, args.learning_rate, args.no_sample_k, args.sample_k, args.provenance, args.top_k)
  trainer.train(args.n_epochs)
