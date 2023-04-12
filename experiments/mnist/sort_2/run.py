import os
import random
from typing import *

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from tqdm import tqdm

import scallopy

class MNISTSort2Dataset(torch.utils.data.Dataset):
  mnist_img_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
      (0.1307,), (0.3081,)
    )
  ])

  def __init__(
    self,
    root: str,
    train: bool = True,
    target_transform: Optional[Callable] = None,
    download: bool = False,
    max_digit: Optional[int] = None,
  ):
    # Contains a MNIST dataset
    self.mnist_dataset = torchvision.datasets.MNIST(
      root,
      train=train,
      transform=self.mnist_img_transform,
      target_transform=target_transform,
      download=download,
    )
    self.index_map = list(range(len(self.mnist_dataset)))
    random.shuffle(self.index_map)

    # Check if we want limited labels
    if max_digit is not None:
      self.index_map = [i for i in self.index_map if self.mnist_dataset[i][1] <= max_digit]

  def __len__(self):
    return int(len(self.index_map) / 2)

  def __getitem__(self, idx):
    # Get two data points
    (a_img, a_digit) = self.mnist_dataset[self.index_map[idx * 2]]
    (b_img, b_digit) = self.mnist_dataset[self.index_map[idx * 2 + 1]]

    # Each data has two images and the GT is the comparison result of two digits
    return (a_img, b_img, 0 if a_digit < b_digit else 1)

  @staticmethod
  def collate_fn(batch):
    a_imgs = torch.stack([item[0] for item in batch])
    b_imgs = torch.stack([item[1] for item in batch])
    cmp = torch.stack([torch.tensor(item[2]).long() for item in batch])
    return ((a_imgs, b_imgs), cmp)


def mnist_sort_2_loader(data_dir, batch_size, max_digit):
  train_loader = torch.utils.data.DataLoader(
    MNISTSort2Dataset(
      data_dir,
      train=True,
      download=True,
      max_digit=max_digit,
    ),
    collate_fn=MNISTSort2Dataset.collate_fn,
    batch_size=batch_size,
    shuffle=True,
  )

  test_loader = torch.utils.data.DataLoader(
    MNISTSort2Dataset(
      data_dir,
      train=False,
      download=True,
      max_digit=max_digit,
    ),
    collate_fn=MNISTSort2Dataset.collate_fn,
    batch_size=batch_size,
    shuffle=True,
  )

  return train_loader, test_loader


class MNISTNet(nn.Module):
  def __init__(self, num_classes=10):
    super(MNISTNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
    self.fc1 = nn.Linear(1024, 1024)
    self.fc2 = nn.Linear(1024, num_classes)

  def forward(self, x):
    x = F.max_pool2d(self.conv1(x), 2)
    x = F.max_pool2d(self.conv2(x), 2)
    x = x.view(-1, 1024)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc2(x)
    return F.softmax(x, dim=1)


class MNISTSort2Net(nn.Module):
  def __init__(self, provenance, train_k, test_k, max_digit=9):
    super(MNISTSort2Net, self).__init__()
    self.max_digit = max_digit
    self.num_classes = max_digit + 1

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet(num_classes=self.num_classes)

    # Scallop Context
    self.scl_ctx = scallopy.ScallopContext(provenance=provenance, train_k=train_k, test_k=test_k)
    self.scl_ctx.add_relation("digit_1", int, input_mapping=list(range(self.num_classes)))
    self.scl_ctx.add_relation("digit_2", int, input_mapping=list(range(self.num_classes)))
    self.scl_ctx.add_rule("less_than(a < b) = digit_1(a), digit_2(b)")

    # The `less_than` logical reasoning module
    self.less_than = self.scl_ctx.forward_function("less_than", [True, False])

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
    (a_imgs, b_imgs) = x

    # First recognize the two digits
    a_distrs = self.mnist_net(a_imgs)
    b_distrs = self.mnist_net(b_imgs)

    # Then execute the reasoning module; the result is a size 2 tensor
    return self.less_than(digit_1=a_distrs, digit_2=b_distrs)


def bce_loss(output, ground_truth):
  (_, dim) = output.shape
  gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
  return F.binary_cross_entropy(output, gt)


def nll_loss(output, ground_truth):
  return F.nll_loss(output, ground_truth)


class Trainer():
  def __init__(self, train_loader, test_loader, model_dir, learning_rate, loss, train_k, test_k, provenance, max_digit=9):
    self.network = MNISTSort2Net(provenance, train_k=train_k, test_k=test_k, max_digit=max_digit)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.model_dir = model_dir
    self.min_test_loss = 100000000.0
    if loss == "nll": self.loss = nll_loss
    elif loss == "bce": self.loss = bce_loss
    else: raise Exception(f"Unknown loss function `{loss}`")

  def train(self, epoch):
    self.network.train()
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    for (data, target) in iter:
      self.optimizer.zero_grad()
      output = self.network(data)
      loss = self.loss(output, target)
      loss.backward()
      self.optimizer.step()
      iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}")

  def test(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    test_loss = 0
    correct = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (data, target) in iter:
        output = self.network(data)
        test_loss += self.loss(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        perc = 100. * correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")
    if test_loss < self.min_test_loss:
      self.min_test_loss = test_loss
      torch.save(self.network, os.path.join(self.model_dir, "mnist_sort_2_net.pkl"))

  def run(self, n_epochs):
    self.test(0)
    for epoch in range(1, n_epochs + 1):
      self.train(epoch)
      self.test(epoch)


if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sort_2")
  parser.add_argument("--n-epochs", type=int, default=10)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--train-k", type=int, default=3)
  parser.add_argument("--test-k", type=int, default=3)
  parser.add_argument("--max-digit", type=int, default=9)
  args = parser.parse_args()

  # Parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Directories
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../model/mnist_sort_2"))
  if not os.path.exists(model_dir): os.mkdir(model_dir)

  # Dataloaders
  train_loader, test_loader = mnist_sort_2_loader(data_dir, args.batch_size, max_digit=args.max_digit)

  # Create trainer and train
  trainer = Trainer(train_loader, test_loader, model_dir, args.learning_rate, args.loss_fn, args.train_k, args.test_k, args.provenance, max_digit=args.max_digit)
  trainer.run(args.n_epochs)
