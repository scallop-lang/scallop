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

mnist_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class MNISTSort2Dataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
    augmentation: int = 2
  ):
    # Contains a MNIST dataset
    self.mnist_dataset = torchvision.datasets.MNIST(
      root,
      train=train,
      transform=transform,
      target_transform=target_transform,
      download=download,
    )
    self.augmentation = augmentation
    self.index_map = list(range(len(self.mnist_dataset))) * augmentation
    random.shuffle(self.index_map)

  def __len__(self):
    return int(len(self.mnist_dataset) / 2) * self.augmentation

  def __getitem__(self, idx):
    # Get two data points
    (a_img, a_digit) = self.mnist_dataset[self.index_map[idx * 2]]
    (b_img, b_digit) = self.mnist_dataset[self.index_map[idx * 2 + 1]]

    # Each data has two images and the GT is the sum of two digits
    return (a_img, b_img, 0 if a_digit < b_digit else 1)

  @staticmethod
  def collate_fn(batch):
    a_imgs = torch.stack([item[0] for item in batch])
    b_imgs = torch.stack([item[1] for item in batch])
    cmp = torch.stack([torch.tensor(item[2]).long() for item in batch])
    return ((a_imgs, b_imgs), cmp)


def mnist_sort_2_loader(data_dir, augmentation, batch_size_train, batch_size_test):
  train_loader = torch.utils.data.DataLoader(
    MNISTSort2Dataset(
      data_dir,
      train=True,
      download=True,
      transform=mnist_img_transform,
      augmentation=augmentation,
    ),
    collate_fn=MNISTSort2Dataset.collate_fn,
    batch_size=batch_size_train,
    shuffle=True
  )

  test_loader = torch.utils.data.DataLoader(
    MNISTSort2Dataset(
      data_dir,
      train=False,
      download=True,
      transform=mnist_img_transform,
      augmentation=augmentation,
    ),
    collate_fn=MNISTSort2Dataset.collate_fn,
    batch_size=batch_size_test,
    shuffle=True
  )

  return train_loader, test_loader


class MNISTNet(nn.Module):
  def __init__(self):
    super(MNISTNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
    self.fc1 = nn.Linear(1024, 1024)
    self.fc2 = nn.Linear(1024, 10)

  def forward(self, x):
    x = F.max_pool2d(self.conv1(x), 2)
    x = F.max_pool2d(self.conv2(x), 2)
    x = x.view(-1, 1024)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc2(x)
    return F.softmax(x, dim=1)


class MNISTSort2Net(nn.Module):
  def __init__(self, provenance, train_k, test_k):
    super(MNISTSort2Net, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()

    # Scallop Context
    self.scl_ctx = scallopy.ScallopContext(provenance=provenance, train_k=train_k, test_k=test_k)
    self.scl_ctx.add_relation("digit_1", "i8", input_mapping=list(range(10)))
    self.scl_ctx.add_relation("digit_2", "i8", input_mapping=list(range(10)))
    self.scl_ctx.add_rule("less_than(a < b) = digit_1(a), digit_2(b)")

    # The `sum_2` logical reasoning module
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
  def __init__(self, train_loader, test_loader, model_dir, model_file, learning_rate, loss, train_k, test_k, provenance):
    self.network = MNISTSort2Net(provenance, train_k=train_k, test_k=test_k)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.model_dir = model_dir
    self.model_file = model_file
    self.min_test_loss = 100000000.0
    if loss == "nll": self.loss = nll_loss
    elif loss == "bce": self.loss = bce_loss
    else: raise Exception(f"Unknown loss function `{loss}`")

  def train_epoch(self, epoch):
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
      torch.save(self.network, os.path.join(self.model_dir, self.model_file))

  def train(self, n_epochs):
    self.test(0)
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test(epoch)


if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sort_2")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--train-k", type=int, default=3)
  parser.add_argument("--test-k", type=int, default=3)
  parser.add_argument("--model-file", type=str, default="mnist_sort_2_net.augmented.pkl")
  parser.add_argument("--augmentation", type=int, default=2)
  args = parser.parse_args()

  # Parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Directories
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../model/mnist_sort_2"))
  if not os.path.exists(model_dir): os.mkdir(model_dir)

  # Dataloaders
  train_loader, test_loader = mnist_sort_2_loader(data_dir, args.augmentation, args.batch_size, args.batch_size)

  # Create trainer and train
  trainer = Trainer(train_loader, test_loader, model_dir, args.model_file, args.learning_rate, args.loss_fn, args.train_k, args.test_k, args.provenance)
  trainer.train(args.n_epochs)
