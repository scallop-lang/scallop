import scallopy
import os

from .config import SAVE_IMAGE_PATH


@scallopy.foreign_function
def save_image(img_tensor: scallopy.Tensor, img_name: str = None) -> str:
    from PIL import Image

    # Convert image
    img = Image.fromarray(img_tensor.numpy())

    # Default name
    if img_name is None:
        img_name = id(img)

    # First get the directory
    directory = SAVE_IMAGE_PATH
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = os.path.join(directory, f"{img_name}.jpg")

    # Save the image
    try:
        img.save(open(file_name, "w"))
        return file_name
    except:
        return ""
