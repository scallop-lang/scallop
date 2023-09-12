import scallopy
import re

@scallopy.foreign_function
def crop_image(
  img: scallopy.Tensor,
  bbox_x: scallopy.u32,
  bbox_y: scallopy.u32,
  bbox_w: scallopy.u32,
  bbox_h: scallopy.u32,
  loc: str = None,
) -> scallopy.Tensor:
  img_h, img_w, _ = img.shape

  # Clip bounding box
  x1, y1, x2, y2 = min(bbox_x, img_w - 1), min(bbox_y, img_h - 1), min(bbox_x + bbox_w, img_w), min(bbox_y + bbox_h, img_h)
  cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

  # Determine crop region
  if loc is not None:
    if loc.startswith("enlarge") or loc.startswith("expand"):
      matches = re.findall("\d*\.*\d+", loc)
      if len(matches) > 0:
        enlarge_box_factor = float(matches[0])
        w, h = int((enlarge_box_factor - 1) * (x2 - x1) / 2), int((enlarge_box_factor - 1) * (y2 - y1) / 2)
        x1, y1, x2, y2 = max(0, x1 - w), max(0, y1 - h), min(img_w, x2 + w), min(img_h, y2 + h)
    elif loc == "left":
      x1, y1, x2, y2 = 0, 0, cx, img_h
    elif loc == "right":
      x1, y1, x2, y2 = cx, 0, img_w, img_h
    elif loc == "above":
      x1, y1, x2, y2 = 0, 0, img_w, cy
    elif loc == "below":
      x1, y1, x2, y2 = 0, cy, img_w, img_h

  return img[y1:y2, x1:x2, :]
