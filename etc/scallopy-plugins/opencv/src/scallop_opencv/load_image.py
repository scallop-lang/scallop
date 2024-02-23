import scallopy


@scallopy.foreign_function
def load_image(image_dir: str) -> scallopy.Tensor:
    # Import
    import torch
    import numpy
    from PIL import Image

    # Load image
    image = Image.open(image_dir).convert("RGB")
    image_tensor = torch.tensor(numpy.asarray(image))

    # Return
    return image_tensor


@scallopy.foreign_function
def load_image_url(image_url: str) -> scallopy.Tensor:
    import torch
    import numpy
    import requests
    from io import BytesIO
    from PIL import Image

    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image_tensor = torch.tensor(numpy.asarray(image))

    return image_tensor
