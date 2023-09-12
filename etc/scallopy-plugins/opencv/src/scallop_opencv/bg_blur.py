import scallopy

"""Code adapted from https://github.com/allenai/visprog.git"""
@scallopy.foreign_function
def bg_blur(img_tensor: scallopy.Tensor, mask_tensor: scallopy.Tensor) -> scallopy.Tensor:
  from PIL import Image, ImageFilter
  import torch
  import numpy as np
  import cv2

  # Process image
  img = Image.fromarray(img_tensor.numpy())
  bg_blur_img = img.copy()
  bg_blur_img = bg_blur_img.filter(ImageFilter.GaussianBlur(radius = 5))
  img, bg_blur_img = np.array(img).astype(float), np.array(bg_blur_img).astype(float)

  # Refine the mask
  mask = np.ascontiguousarray(mask_tensor)
  mask, _, _ = cv2.grabCut(
    img.astype(np.uint8),
    mask.astype(np.uint8),
    None,
    np.zeros((1,65),np.float64),
    np.zeros((1,65),np.float64),
    5,
    cv2.GC_INIT_WITH_MASK
  )
  mask = mask.astype(float)
  mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))

  # Apply Gaussian blur mask
  mask = Image.fromarray(255 * mask.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius = 5))
  mask = np.array(mask).astype(float) / 255
  bg_blur_img = mask * img + (1 - mask) * bg_blur_img

  return torch.tensor(np.asarray(bg_blur_img).astype(np.uint8))
