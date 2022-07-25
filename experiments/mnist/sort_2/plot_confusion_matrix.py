from argparse import ArgumentParser
import os

# Computation
import torch
import torchvision
import numpy
from sklearn.metrics import confusion_matrix

# Plotting
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from run import MNISTSort2Dataset, MNISTNet, MNISTSort2Net

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("plot_confusion_matrix")
  parser.add_argument("--model-file", default="mnist_sort_2/mnist_sort_2_net.pkl")
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--plot-image", action="store_true")
  parser.add_argument("--image-file", default="confusion.png")
  args = parser.parse_args()

  # Directories
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../model"))

  # Load mnist dataset
  mnist_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=MNISTSort2Dataset.mnist_img_transform)
  mnist_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=args.batch_size)

  # Load model
  mnist_sort_2_net: MNISTSort2Net = torch.load(open(os.path.join(model_dir, args.model_file), "rb"))
  mnist_net: MNISTNet = mnist_sort_2_net.mnist_net
  mnist_net.eval()

  # Get prediction result
  y_true, y_pred = [], []
  with torch.no_grad():
    for (imgs, digits) in mnist_loader:
      pred_digits = numpy.argmax(mnist_net(imgs), axis=1)
      mask = digits <= mnist_sort_2_net.max_digit
      y_true += [d.item() for (i, d) in enumerate(digits) if mask[i]]
      y_pred += [d.item() for (i, d) in enumerate(pred_digits) if mask[i]]

  # Compute confusion matrix
  cm = confusion_matrix(y_true, y_pred)

  # Plot image or print
  if args.plot_image:
    df_cm = pd.DataFrame(cm, index=list(range(10)), columns=list(range(10)))
    plt.figure(figsize=(10,7))
    sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(args.image_file)
  else:
    print(cm)
