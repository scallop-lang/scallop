from argparse import ArgumentParser
import random
import os

import torch

from run_with_augmentation import MNISTSort2Dataset

if __name__ == "__main__":
  parser = ArgumentParser("mnist_sort_2.dataset_stats")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--augmentation", type=int, default=2)
  args = parser.parse_args()

  # Parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Directories
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))

  # Store
  store = torch.zeros(10, 10).long()

  # Dataloaders
  train_dataset = MNISTSort2Dataset(data_dir, train=True, download=True, augmentation=args.augmentation)
  for i in range(len(train_dataset)):
    (_, a_digit) = train_dataset.mnist_dataset[train_dataset.index_map[i * 2]]
    (_, b_digit) = train_dataset.mnist_dataset[train_dataset.index_map[i * 2 + 1]]
    store[a_digit, b_digit] += 1

  # Print store
  print(store)
