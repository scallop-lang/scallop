import scallopy

# For memoization
STORAGE = {}

@scallopy.foreign_function
def load_image(image_dir: str) -> scallopy.Tensor:
  # Import
  import torch
  import numpy
  from PIL import Image

  # Load image
  image = Image.open(image_dir).convert('RGB')
  image_tensor = torch.tensor(numpy.asarray(image))

  # Return
  return image_tensor
