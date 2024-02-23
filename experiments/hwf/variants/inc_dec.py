import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import json
import scallopy
import random
import os
from PIL import Image
from tqdm import tqdm
from typing import *

data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))

class ECExprDataset(torch.utils.data.Dataset):
  def __init__(self, train: bool = True):
    split = "train" if train else "test"
    self.metadata = json.load(open(os.path.join(data_dir, f"HWF/inc_dec_expr_{split}.json")))
    self.img_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (1,))])

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx):
    datapoint = self.metadata[idx]
    imgs = []
    for img_path in datapoint["img_paths"]:
      img_full_path = os.path.join(data_dir, "HWF/Handwritten_Math_Symbols", img_path)
      img = Image.open(img_full_path).convert("L")
      img = self.img_transform(img)
      imgs.append(img)
    res = datapoint["res"]
    return (tuple(imgs), res)

def EC_expr_loader(batch_size):
  train_loader = torch.utils.data.DataLoader(ECExprDataset(train=True), batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(ECExprDataset(train=False), batch_size=batch_size, shuffle=True)
  return train_loader, test_loader

class ConvolutionNeuralNetEC(nn.Module):
  def __init__(self, num_classes):
    super(ConvolutionNeuralNetEC, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, stride = 1, padding = 1)
    self.conv2 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
    self.fc1 = nn.Linear(7744, 128)
    self.fc2 = nn.Linear(128, num_classes)

  def forward(self, x):
    x = F.max_pool2d(self.conv1(x), 2)
    x = F.max_pool2d(self.conv2(x), 2)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p = 0.5, training=self.training)
    x = self.fc2(x)
    return F.softmax(x, dim=1)

class ECExprNet(nn.Module):
  def __init__(self):
    super(ECExprNet, self).__init__()

    # Symbol Recognition CNN(s)
    self.digit_cnn = ConvolutionNeuralNetEC(10)
    self.symbol_cnn = ConvolutionNeuralNetEC(2)

    # Scallop Context
    self.compute = scallopy.ScallopForwardFunction(
      program="""
      type digit(i32), op1(String), op2(String)
      rel result(a + 1) = digit(a) and op1("+") and op2("+")
      rel result(a - 1) = digit(a) and op1("-") and op2("-")
      """,
      provenance="difftopkproofs",
      input_mappings={
        "digit": list(range(10)),
        "op1": ["+", "-"],
        "op2": ["+", "-"],
      },
      output_relation="result",
      output_mapping=list(range(-1, 11)),
    )

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
    return self.compute(
      digit=self.digit_cnn(x[0]),
      op1=self.symbol_cnn(x[1]),
      op2=self.symbol_cnn(x[2]),
    )

class ECExprTrainer():
  def __init__(self, train_loader, test_loader, learning_rate):
    self.network = ECExprNet()
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader

  def loss(self, output, ground_truth):
    output_mapping = list(range(-1, 11))
    (_, dim) = output.shape
    gt = torch.stack([torch.tensor([1.0 if output_mapping[i] == t else 0.0 for i in range(dim)]) for t in ground_truth])
    return F.binary_cross_entropy(output, gt)

  def train_epoch(self, epoch):
    self.network.train()
    train_loss = 0.0
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    for (batch_id, (data, target)) in enumerate(iter):
      self.optimizer.zero_grad()
      output = self.network(data)
      loss = self.loss(output, target)
      loss.backward()
      self.optimizer.step()
      train_loss += loss.item()
      avg_loss = train_loss / (batch_id + 1)
      iter.set_description(f"[Train Epoch {epoch}] Batch Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = 0
    test_loss = 0
    correct = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (batch_id, (data, target)) in enumerate(iter):
        output = self.network(data)
        test_loss += self.loss(output, target).item()
        avg_loss = test_loss / (batch_id + 1)
        pred = output.data.max(1, keepdim=True)[1] - 1
        correct += pred.eq(target.data.view_as(pred)).sum()
        num_items += pred.shape[0]
        perc = 100. * correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Avg loss: {avg_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")

  def train(self, n_epochs):
    self.test_epoch(0)
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)

# Parameters
n_epochs = 3
batch_size = 32
learning_rate = 0.001
seed = 1234

# Random seed
torch.manual_seed(seed)
random.seed(seed)

# Dataloaders
train_loader, test_loader = EC_expr_loader(batch_size)
trainer = ECExprTrainer(train_loader, test_loader, learning_rate)
trainer.train(n_epochs)
