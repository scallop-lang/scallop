import scallopy
import os


@scallopy.foreign_function
def upload_imgur(img_tensor: scallopy.Tensor) -> str:
    from .save_image import save_image

    img_path = save_image(img_tensor)

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
