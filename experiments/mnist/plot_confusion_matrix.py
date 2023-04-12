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

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("plot_confusion_matrix")
  parser.add_argument("--model-file", default="mnist_sort_2/mnist_sort_2_net.pkl")
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--plot-image", action="store_true")
  parser.add_argument("--image-file", default="confusion.png")
  parser.add_argument("--task", type=str, default="sum_2")
  args = parser.parse_args()

  if args.task == "sum_2":
    import sum_2 as module
    from sum_2 import MNISTSum2Net, MNISTNet

  # Directories
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model"))

  # Load mnist dataset
  mnist_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=module.mnist_img_transform)
  mnist_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=args.batch_size)

  # Load model
  mnist_subtask_net = torch.load(open(os.path.join(model_dir, args.model_file), "rb"))
  mnist_net = mnist_subtask_net.mnist_net
  mnist_net.eval()

  # Get prediction result
  y_true, y_pred = [], []
  with torch.no_grad():
    for (imgs, digits) in mnist_loader:
      pred_digits = numpy.argmax(mnist_net(imgs), axis=1)
      y_true += [d.item() for d in digits]
      y_pred += [d.item() for d in pred_digits]

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
