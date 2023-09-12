import scallopy

"""Code adapted from https://github.com/allenai/visprog.git"""
@scallopy.foreign_function
def color_pop(img_tensor: scallopy.Tensor, mask_tensor: scallopy.Tensor) -> scallopy.Tensor:
  from PIL import Image
  import torch
  import numpy as np
  import cv2

  # Process image
  img = Image.fromarray(img_tensor.numpy())
  color_pop_img = img.copy()
  color_pop_img = color_pop_img.convert('L').convert('RGB')
  img, color_pop_img = np.array(img).astype(float), np.array(color_pop_img).astype(float)

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

  # Apply color pop mask
  color_pop_img = mask * img + (1 - mask) * color_pop_img

  return torch.tensor(np.asarray(color_pop_img).astype(np.uint8))
