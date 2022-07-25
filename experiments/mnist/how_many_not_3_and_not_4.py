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

class MNISTHowMany3Or4Dataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    num_elements: int = 3,
    dataset_up_scale: int = 2,
    download: bool = False,
  ):
    # Contains a MNIST dataset
    self.mnist_dataset = torchvision.datasets.MNIST(
      root,
      train=train,
      transform=transform,
      target_transform=target_transform,
      download=download,
    )
    self.num_elements = num_elements
    self.dataset_up_scale = dataset_up_scale
    self.index_map = list(range(len(self.mnist_dataset))) * self.dataset_up_scale
    random.shuffle(self.index_map)

  def __len__(self):
    return int(len(self.index_map) / self.num_elements)

  def __getitem__(self, idx):
    # Get data points
    img_and_digits = [self.mnist_dataset[self.index_map[idx * self.num_elements + offset]] for offset in range(self.num_elements)]

    # Digits
    digits = [d for (_, d) in img_and_digits]
    how_many = len([() for d in digits if d != 3 and d != 4])

    # Images
    images = [img for (img, _) in img_and_digits]

    # Each data has two images and the GT is the sum of two digits
    return (images, how_many)

  @staticmethod
  def collate_fn(batch):
    num_elements = len(batch[0][0])
    imgs = [torch.stack([item[0][i] for item in batch]) for i in range(num_elements)]
    answer = torch.stack([torch.tensor(item[1]).long() for item in batch])
    return (imgs, answer)


def mnist_how_many_3_or_4_loader(data_dir, num_elements, dataset_up_scale, batch_size_train, batch_size_test):
  train_loader = torch.utils.data.DataLoader(
    MNISTHowMany3Or4Dataset(
      data_dir,
      train=True,
      download=True,
      transform=mnist_img_transform,
      num_elements=num_elements,
      dataset_up_scale=dataset_up_scale,
    ),
    collate_fn=MNISTHowMany3Or4Dataset.collate_fn,
    batch_size=batch_size_train,
    shuffle=True
  )

  test_loader = torch.utils.data.DataLoader(
    MNISTHowMany3Or4Dataset(
      data_dir,
      train=False,
      download=True,
      transform=mnist_img_transform,
      num_elements=num_elements,
      dataset_up_scale=dataset_up_scale,
    ),
    collate_fn=MNISTHowMany3Or4Dataset.collate_fn,
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
    x = F.dropout(x, p = 0.5, training=self.training)
    x = self.fc2(x)
    return F.softmax(x, dim=1)


class MNISTHowMany3Or4Net(nn.Module):
  def __init__(self, num_elements, provenance, k):
    super(MNISTHowMany3Or4Net, self).__init__()
    self.num_elements = num_elements

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()

    # Scallop Context
    self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    self.scl_ctx.add_relation("all_digits", int, non_probabilistic=True)
    self.scl_ctx.add_relation("digit", (int, int), input_mapping=[(i, d) for i in range(num_elements) for d in range(10)])
    self.scl_ctx.add_rule("how_many(x) :- x = count(o: all_digits(o) and ~digit(o, 3) and ~digit(o, 4))")

    # The `how_many` logical reasoning module
    self.how_many = self.scl_ctx.forward_function("how_many", list(range(num_elements + 1)))

  def forward(self, x: List[torch.Tensor]):
    batch_size = len(x[0])

    # Apply mnist net on each image
    digit_distrs = [self.mnist_net(imgs) for imgs in x]

    # Concatenate them into the same big tensor
    digit = torch.cat(tuple(digit_distrs), dim=1)

    # Get the all_digits
    all_digits = [[(j,) for j in range(self.num_elements)] for _ in range(batch_size)]

    # Then execute the reasoning module; the result is a size `num_elements + 1` tensor
    return self.how_many(digit=digit, all_digits=all_digits)


def bce_loss(output, ground_truth):
  (_, dim) = output.shape
  gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
  return F.binary_cross_entropy(output, gt)


def nll_loss(output, ground_truth):
  return F.nll_loss(output, ground_truth)


class Trainer():
  def __init__(self, train_loader, test_loader, learning_rate, loss, num_elements, k, provenance):
    self.network = MNISTHowMany3Or4Net(num_elements, provenance, k)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    if loss == "nll":
      self.loss = nll_loss
    elif loss == "bce":
      self.loss = bce_loss
    else:
      raise Exception(f"Unknown loss function `{loss}`")

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

  def test_epoch(self, epoch):
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

  def train(self, n_epochs):
    self.test_epoch(0)
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)


if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_how_many_3")
  parser.add_argument("--n-epochs", type=int, default=10)
  parser.add_argument("--batch-size-train", type=int, default=64)
  parser.add_argument("--batch-size-test", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--num-elements", type=int, default=3)
  parser.add_argument("--dataset-up-scale", type=int, default=2)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--top-k", type=int, default=2)
  args = parser.parse_args()

  # Parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))

  # Dataloaders
  train_loader, test_loader = mnist_how_many_3_or_4_loader(data_dir, args.num_elements, args.dataset_up_scale, args.batch_size_train, args.batch_size_test)

  # Create trainer and train
  trainer = Trainer(train_loader, test_loader, args.learning_rate, args.loss_fn, args.num_elements, args.top_k, args.provenance)
  trainer.train(args.n_epochs)
