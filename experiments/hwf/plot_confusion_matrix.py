from argparse import ArgumentParser
import os
import random

# Computation
import torch
import torchvision
import numpy
import functools
from sklearn.metrics import confusion_matrix
from PIL import Image

# Plotting
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from run_with_hwf_parser import *

class HWFSymbolDataset(torch.utils.data.Dataset):
  def __init__(self, root: str, amount: int = 1000):
    super(HWFSymbolDataset, self).__init__()
    self.root = root
    self.img_transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.5,), (1,))
    ])
    self.amount = amount
    self.data = []
    self.symbols = [str(i) for i in range(10)] + ["+", "-", "*", "/"]
    self.data = []
    for _ in range(self.amount):
      r1 = random.randint(0, 13)
      if r1 < 10: images = self.images_of_digit(r1)
      else: images = self.images_of_symbol(self.symbols[r1])
      r2 = random.randint(0, len(images) - 1)
      img_path = images[r2]
      self.data.append((img_path, r1))

  @functools.lru_cache
  def images_of_digit(self, digit: int):
    return [f"{digit}/{f}" for f in os.listdir(os.path.join(self.root, "HWF/Handwritten_Math_Symbols", str(digit)))]

  @functools.lru_cache
  def images_of_symbol(self, symbol: str):
    if symbol == "+": d = "+"
    elif symbol == "-": d = "-"
    elif symbol == "*": d = "times"
    elif symbol == "/": d = "div"
    else: raise Exception(f"Unknown symbol {symbol}")
    return [f"{d}/{f}" for f in os.listdir(os.path.join(self.root, "HWF/Handwritten_Math_Symbols", d))]

  def __len__(self):
    return self.amount

  def __getitem__(self, index):
    (img_path, label) = self.data[index]
    img_full_path = os.path.join(self.root, "HWF/Handwritten_Math_Symbols", img_path)
    img = Image.open(img_full_path).convert("L")
    img = self.img_transform(img)
    return (img, label)


if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("plot_confusion_matrix")
  parser.add_argument("--amount", type=int, default=1000)
  parser.add_argument("--model-name", default="hwf/hwf.pkl")
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--plot-image", action="store_true")
  parser.add_argument("--image-file", default="confusion.png")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--accuracy", action="store_true")
  args = parser.parse_args()

  # Directories
  random.seed(args.seed)
  torch.manual_seed(args.seed)
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model"))

  # Load mnist dataset
  dataset = HWFSymbolDataset(data_dir, amount=args.amount)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

  # Load model
  hwf_net = torch.load(open(os.path.join(model_dir, args.model_name), "rb"))
  symbol_net: SymbolNet = hwf_net.symbol_cnn
  symbol_net.eval()

  # Get prediction result
  y_true, y_pred = [], []
  with torch.no_grad():
    for (imgs, digits) in dataloader:
      pred_digits = numpy.argmax(symbol_net(imgs), axis=1)
      y_true += [d.item() for d in digits]
      y_pred += [d.item() for d in pred_digits]

  # Compute accuracy if asked
  if args.accuracy:
    acc = float(len([() for (x, y) in zip(y_true, y_pred) if x == y and x != 0])) / float(len([() for x in y_true if x != 0]))
    print(f"Accuracy: {acc:4f}")

  # Compute confusion matrix
  cm = confusion_matrix(y_true, y_pred)

  # Plot image or print
  if args.plot_image:
    df_cm = pd.DataFrame(cm, index=list(range(14)), columns=list(range(14)))
    plt.figure(figsize=(10,7))
    sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(args.image_file)
  else:
    print(cm)
