import scallopy
import os

from .save_image import _save_image

def _upload_imgur(
    img_tensor: scallopy.Tensor,
    save_image_path: str = "",
) -> str:
  img_path = _save_image(img_tensor, save_image_path=save_image_path)

  if len(img_path) == 0:
    return ""

  client_id = os.environ.get("IMGUR_API_ID")
  client_secret = os.environ.get("IMGUR_API_SECRET")
  if client_id is None or client_secret is None:
    return ""

  from imgurpython import ImgurClient

  client = ImgurClient(client_id, client_secret)
  response = client.upload_from_path(img_path)

  return response["link"]
