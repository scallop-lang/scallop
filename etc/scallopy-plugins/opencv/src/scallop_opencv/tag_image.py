import scallopy
import os

"""Code adapted from https://github.com/allenai/visprog.git"""
@scallopy.foreign_function
def tag_image(
  img_tensor: scallopy.Tensor,
  bbox_x: scallopy.u32,
  bbox_y: scallopy.u32,
  bbox_w: scallopy.u32,
  bbox_h: scallopy.u32,
  label: str = None,
  color: str = None,
  width: int = None,
  font_size: int = None,
  enlarge_factor: float = None,
) -> scallopy.Tensor:
  from PIL import Image, ImageDraw, ImageFont
  import torch
  import numpy

  # Process image
  img = Image.fromarray(img_tensor.numpy())
  img_w, img_h = img.size

  # Default color
  if color is None:
    color = "green"

  # Default width
  if width is None:
    width = 4

  # Default font dize
  if font_size is None:
    font_size = 16

  # Default enlarge factor
  if enlarge_factor is None:
    enlarge_factor = 1

  # Enlarge bounding box
  w, h = int(enlarge_factor * bbox_w), int(enlarge_factor * bbox_h)
  x, y = bbox_x - int((w - bbox_w) / 2), bbox_y - int((h - bbox_h) / 2)

  # Clip bounding box
  x1, y1, x2, y2 = min(x, img_w - 1), min(y, img_h - 1), min(x + w, img_w), min(y + h, img_h)

  # Draw the box
  tagged_img = img.copy()
  draw = ImageDraw.Draw(tagged_img)
  draw.rectangle((x1, y1, x2, y2), outline=color, width=width)

  # Draw the label
  if label is not None:
    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../res/testing/fonts/NotoSansMono-Bold.ttf")
    font = ImageFont.truetype(font_path, font_size)
    fw, fh = font.getsize(label)
    if x1 + fw > img_w:
      x1 -= fw - (x2 - x1)
    if y2 + fh > img_h:
      y2 -= fh
    draw.rectangle((x1, y2, x1 + fw, y2 + fh), fill=color)
    draw.text((x1, y2), label, fill='white', font=font)

  return torch.tensor(numpy.asarray(tagged_img))
