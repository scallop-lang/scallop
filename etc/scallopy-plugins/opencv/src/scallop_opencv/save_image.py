import scallopy
import os

def _save_image(
    img_tensor: scallopy.Tensor,
    img_name: str = None,
    save_image_path: str = "",
) -> str:
  from PIL import Image

  # Convert image
  img = Image.fromarray(img_tensor.numpy())

  # Default name
  if img_name is None:
    img_name = id(img)

  # First get the directory
  directory = save_image_path
  if not os.path.exists(directory):
    os.makedirs(directory)
  file_name = os.path.join(directory, f"{img_name}.jpg")

  # Save the image
  try:
    img.save(open(file_name, "w"))
    return file_name
  except:
    return ""
