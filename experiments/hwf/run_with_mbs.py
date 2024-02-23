import os
import json
import random
from argparse import ArgumentParser
from tqdm import tqdm
from queue import PriorityQueue
import math

import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from PIL import Image

import scallopy

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
  train_loader = torch.utils.data.DataLoader(
    HWFDataset(data_dir, prefix, "train"),
    collate_fn=HWFDataset.collate_fn,
    batch_size=batch_size,
    shuffle=True,
  )
  test_loader = torch.utils.data.DataLoader(
    HWFDataset(data_dir, prefix, "test"),
    collate_fn=HWFDataset.collate_fn,
    batch_size=batch_size,
    shuffle=True,
  )
  return (train_loader, test_loader)


class ASTLeaf:
  def __init__(self, prob, id, symbol):
    self.prob = prob
    self.id = id
    self.symbol = symbol

  def node_id(self):
    return self.id

  def is_operator(self):
    return self.symbol in ["+", "-", "*", "/"]

  def contains(self, other):
    if isinstance(other, ASTLeaf): return self.id == other.id and self.symbol == other.symbol
    else: return False

  def probability(self):
    return self.prob

  def __repr__(self):
    return f"{self.symbol}"

  def __eq__(self, other):
    if isinstance(other, ASTLeaf): return self.id == other.id and self.symbol == other.symbol
    else: return False


class ASTNode:
  def __init__(self, lhs, op, rhs):
    self.lhs = lhs
    self.op = op
    self.rhs = rhs

  def node_id(self):
    return self.op.node_id()

  def is_operator(self):
    return False

  def contains(self, other):
    if isinstance(other, ASTNode):
      return self == other or self.lhs.contains(other) or self.rhs.contains(other)
    else:
      return self.lhs.contains(other) or self.rhs.contains(other)

  def probability(self):
    return self.lhs.probability() * self.op.probability() * self.rhs.probability()

  def __repr__(self):
    return f"({self.lhs} {self.op} {self.rhs})"

  def __eq__(self, other):
    if isinstance(other, ASTNode): return self.lhs == other.lhs and self.op == other.op and self.rhs == other.rhs
    else: return False


class ASTTag:
  def __init__(self, asts):
    deduped = []
    for ast in asts:
      if ast not in deduped:
        deduped.append(ast)
    self.asts = deduped

  def probability(self):
    if len(self.asts) == 0:
      return 0.0
    else:
      for ast in self.asts:
        if isinstance(ast, ASTNode) or isinstance(ast, ASTLeaf):
          return ast.probability()

  def __repr__(self):
    return f"asttag({self.asts})"

  def filter_valid_proofs(self):
    self.asts = [ast for ast in self.asts if type(ast) != list]

  def one_bs(self, expected):
    self.filter_valid_proofs()
    for ast in self.asts:
      q = PriorityQueue() # Priority queue
      q.put((1, (ast, expected)))
      while True:
        (_, (a, alpha_a)) = q.get()
        print(a, alpha_a)
        exit(0)


class MBSSemiring(scallopy.ScallopProvenance):
  def base(self, info):
    if info is not None:
      (prob, id, symbol) = info
      leaf = ASTLeaf(prob, id, symbol)
      return ASTTag([leaf])
    else:
      return self.one()

  def is_valid(self, tag: ASTNode):
    return len(tag.asts) > 0

  def zero(self):
    return ASTTag([])

  def one(self):
    return ASTTag([[]])

  def add(self, t1: ASTTag, t2: ASTTag):
    return ASTTag(t1.asts + t2.asts)

  def mult(self, t1: ASTTag, t2: ASTTag):
    joined_asts = []
    for a1 in t1.asts:
      for a2 in t2.asts:
        if type(a1) == list and type(a2) == list: joined_ast = a1 + a2
        elif type(a1) == list: joined_ast = [a for a in a1 if not a2.contains(a)] + [a2]
        elif type(a2) == list: joined_ast = [a1] + [a for a in a2 if not a1.contains(a)]
        else:
          if a1.contains(a2): joined_ast = [a1]
          elif a2.contains(a1): joined_ast = [a2]
          else: joined_ast = [a1, a2]

        # Join things
        if len(joined_ast) == 3:
          joined_ast = sorted(joined_ast, key=lambda n: n.node_id())
          if joined_ast[1].is_operator():
            joined_ast = ASTNode(joined_ast[0], joined_ast[1], joined_ast[2])
            joined_asts.append(joined_ast)
        elif len(joined_ast) == 1: joined_asts.append(joined_ast[0])
        elif len(joined_ast) < 3: joined_asts.append(joined_ast)
        else: raise Exception("Should not happen")

    # Create the tag
    return ASTTag(joined_asts)

  def aggregate_unique(self, elements):
    max_prob = 0.0
    max_elem = None
    for (tag, tup) in elements:
      tag_prob = tag.probability()
      if tag_prob > max_prob:
        max_prob = tag_prob
        max_elem = (tag, tup)
    if max_elem is not None: return [max_elem]
    else: return []


class SymbolNet(nn.Module):
  def __init__(self):
    super(SymbolNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, stride = 1, padding = 1)
    self.conv2 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
    self.dropout1 = nn.Dropout2d(0.25)
    self.dropout2 = nn.Dropout2d(0.5)
    self.fc1 = nn.Linear(30976, 128)
    self.fc2 = nn.Linear(128, 14)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    return F.softmax(x, dim=1)


class HWFNet(nn.Module):
  def __init__(self):
    super(HWFNet, self).__init__()

    # Symbol embedding
    self.symbol_cnn = SymbolNet()

    # Scallop context
    self.symbols = [str(i) for i in range(10)] + ["+", "-", "*", "/"]
    self.ctx = scallopy.ScallopContext(provenance="custom", custom_provenance=MBSSemiring())
    self.ctx.import_file(os.path.abspath(os.path.join(os.path.abspath(__file__), "../scl/hwf_unique_parser.scl")))
    self.ctx.set_non_probabilistic("length")
    self.eval_formula = self.ctx.forward_function("result")

  def forward(self, img_seq, img_seq_len):
    batch_size, formula_length, _, _, _ = img_seq.shape
    length = [[(l.item(),)] for l in img_seq_len]
    symbol = self.symbol_cnn(img_seq.flatten(start_dim=0, end_dim=1)).view(batch_size, formula_length, -1)
    symbol_facts = [[] for _ in range(batch_size)]
    for task_id in range(batch_size):
      for symbol_id in range(img_seq_len[task_id]):
        symbols_distr = symbol[task_id, symbol_id]
        curr_symbol_facts = [((p, symbol_id, self.symbols[k]), (symbol_id, self.symbols[k])) for (k, p) in enumerate(symbols_distr)]
        symbol_facts[task_id] += curr_symbol_facts
    (result_mapping, tags) = self.eval_formula(symbol=symbol_facts, length=length)
    return self._extract_result(result_mapping, tags, batch_size)

  def _extract_result(self, result_mapping, tags, batch_size):
    result = []
    for task_id in range(batch_size):
      max_prob = 0.0
      max_tag = None
      max_result_id = None
      for i, tag in enumerate(tags[task_id]):
        if tag is not None:
          p = tag.probability()
          if p is not None and p > max_prob:
            max_prob = p
            max_tag = tag
            max_result_id = i
      if max_result_id: result.append((result_mapping[max_result_id][0], max_tag))
      else: result.append(None)
    return result


class Trainer():
  def __init__(self, train_loader, test_loader, device, model_root, model_name, learning_rate):
    self.network = HWFNet().to(device)
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
      self.optimizer.zero_grad()

      # Run the network and get the results
      result = self.network(img_seq.to(device), img_seq_len.to(device))

      for (task_id, (y_pred, tag)) in enumerate(result):
        y = label[task_id]
        tag.one_bs(y)





      # # Normalize label format
      # batch_size, num_outputs = y_pred.shape
      # y = torch.tensor([1.0 if self.eval_result_eq(l.item(), m) else 0.0 for l in label for m in output_mapping]).view(batch_size, -1)

      # # Compute loss
      # loss = self.loss_fn(y_pred, y)
      # train_loss += loss.item()
      # loss.backward()
      # self.optimizer.step()

      # # Collect index and compute accuracy
      # correct_count = 0
      # if num_outputs > 0:
      #   y_index = torch.argmax(y, dim=1)
      #   y_pred_index = torch.argmax(y_pred, dim=1)
      #   correct_count = torch.sum(torch.where(torch.sum(y, dim=1) > 0, y_index == y_pred_index, torch.zeros(batch_size).bool())).item()

      # # Stats
      # num_items += batch_size
      # total_correct += correct_count
      # perc = 100. * total_correct / num_items
      # avg_loss = train_loss / (i + 1)

      # # Prints
      # iter.set_description(f"[Train Epoch {epoch}] Avg loss: {avg_loss:.4f}, Accuracy: {total_correct}/{num_items} ({perc:.2f}%)")

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
  parser.add_argument("--dataset-prefix", type=str, default="expr")
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--cuda", action="store_true")
  parser.add_argument("--gpu", type=int, default=0)
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
  trainer = Trainer(train_loader, test_loader, device, model_dir, args.model_name, args.learning_rate)
  trainer.train(args.n_epochs)
