from typing import List, Optional
import os

from PIL import Image
import numpy
import torch
import scallopy
from segment_anything import SamAutomaticMaskGenerator, SamPredictor

from . import config

FA_NAME = "segment_anything"
ERR_HEAD = f"[@{FA_NAME}]"

DUMP_IMAGE_PATH = ".tmp/scallop-sam/"

FIELDS = {
  "area": {
    "description": "The area under the generated mask",
    "type": scallopy.u32,
  },
  "bbox-x": {
    "description": "The x-coordinate of the bounding box",
    "type": scallopy.u32,
  },
  "bbox-y": {
    "description": "The y-coordinate of the bounding box",
    "type": scallopy.u32,
  },
  "bbox-x-center": {
    "description": "The x-coordinate of the bounding box center",
    "type": scallopy.u32,
  },
  "bbox-y-center": {
    "description": "The y-coordinate of the bounding box center",
    "type": scallopy.u32,
  },
  "bbox-w": {
    "description": "The width of the bounding box",
    "type": scallopy.u32,
  },
  "bbox-h": {
    "description": "The height of the bounding box",
    "type": scallopy.u32,
  },
  "cropped-image": {
    "description": "The cropped image (bounded by bounding box)",
    "type": scallopy.Tensor,
  },
  "cropped-mask": {
    "description": "The cropped mask (bounded by bounding box)",
    "type": scallopy.Tensor,
  },
  "masked-cropped-image": {
    "description": "The cropped image which is also masked (bounded by bounding box)",
    "type": scallopy.Tensor,
  },
  "mask": {
    "description": "The black and white mask",
    "type": scallopy.Tensor,
  },
}


@scallopy.foreign_attribute
def segment_anything(
  item,
  output_fields: List[str],
  *,
  prompt_bb: bool = False,
  iou_threshold: float = 0.88,
  area_threshold: int = 0,
  limit: Optional[int] = None,
  expand_crop_region: int = 0,
  flatten_prob: bool = False,
  debug: bool = False,
  dump_image: bool = False,
):
  # Needs to be a relation declaration, and can have only one relation
  assert item.is_relation_decl(), f"{ERR_HEAD} has to be an attribute of a relation type declaration"
  assert len(item.relation_decls()) == 1, f"{ERR_HEAD} cannot be an attribute on multiple relations"

  # Get the argument names and argument types
  relation_decl = item.relation_decl(0)
  arg_names = [ab.name.name for ab in relation_decl.arg_bindings]
  arg_types = [ab.ty for ab in relation_decl.arg_bindings]

  # Expected number of input arguments
  expected_args = [(scallopy.Tensor, "image: Tensor")]

  # Bounding box prompt
  if prompt_bb:
    expected_args += [(scallopy.u32, "prompt_bb_x: u32"), (scallopy.u32, "prompt_bb_y: u32"), (scallopy.u32, "prompt_bb_w: u32"), (scallopy.u32, "prompt_bb_h: u32")]

  # Expected output arguments
  expected_args += [(scallopy.u32, "id: u32")]

  # Fields: make sure that all output fields are supported
  assert type(output_fields) == list, f"{ERR_HEAD} The parameter `output_fields` need to be a list"
  assert len(output_fields) > 0, f"{ERR_HEAD} The `output_fields` list must be non-empty"
  for field in output_fields:
    assert field in FIELDS, f"{ERR_HEAD} Unknown output field `{field}`; supported fields are {list(FIELDS.keys())}"
  expected_args += [(FIELDS[f]["type"], f + ": " + FIELDS[f]["type"]) for f in output_fields]

  # Check arity of relation
  assert len(arg_types) == len(expected_args), f"{ERR_HEAD} relation has to be of arity-{len(expected_args)}: {[s for (_, s) in expected_args]}"

  # Making sure that arg types are as expected
  for arg_id in range(len(arg_types)):
    (expected_ty, s), arg_name, arg_ty = expected_args[arg_id], arg_names[arg_id], arg_types[arg_id].name()
    if arg_ty.lower() != expected_ty.lower():
      if arg_name is not None: raise Exception(f"{ERR_HEAD} Type mismatch for `{arg_name}`. Expected `{s}`, but found arg of type `{arg_ty}`")
      else: raise Exception(f"{ERR_HEAD} Type mismatch for argument `{arg_id+1}`. Expected `{s}`, but found arg of type `{arg_ty}`")

  # Normal segment anything model without prompting
  @scallopy.foreign_predicate(
    name=relation_decl.name.name,
    input_arg_types=arg_types[:1],
    output_arg_types=arg_types[1:],
    tag_type=float,
  )
  def do_segment(img: scallopy.Tensor):
    sam = config.get_sam_model()
    if sam is not None:
      predictor = SamAutomaticMaskGenerator(
        sam,
        pred_iou_thresh=iou_threshold,
        min_mask_region_area=area_threshold,
      )
      img_w, img_h, _ = img.shape
      masks = predictor.generate(img.numpy())

      # Count
      yielded_count = 0

      # Iterate through all the segments
      for (i, mask) in enumerate(masks):
        if debug: print(mask)

        # Basic information
        prob = mask["predicted_iou"] if not flatten_prob else 0.0
        (bbox_x, bbox_y, bbox_w, bbox_h) = mask["bbox"]
        (x_crop_slice, y_crop_slice) = get_xy_cropping_slices(img_w, img_h, bbox_y, bbox_x, bbox_h, bbox_w, expand_crop_region)
        segmentation = torch.from_numpy(mask["segmentation"])

        # Check yielded count; stop if reaching limit
        if limit is not None and yielded_count > limit: break
        else: yielded_count += 1

        # Get all the information for the fields
        result_tuple = [i]
        for field in output_fields:
          if field == "mask":
            result_tuple.append(segmentation)
            if dump_image:
              save_temporary_image(id(img), i, field, segmentation)
          elif field == "masked-cropped-image":
            masked_img = torch.mul(img, segmentation.unsqueeze(-1))
            masked_cropped_img = masked_img[x_crop_slice, y_crop_slice, :]
            result_tuple.append(masked_cropped_img)
            if dump_image:
              save_temporary_image(id(img), i, field, masked_cropped_img)
          elif field == "cropped-image":
            cropped_img = img[x_crop_slice, y_crop_slice, :]
            result_tuple.append(cropped_img)
            if dump_image:
              save_temporary_image(id(img), i, field, cropped_img)
          elif field == "cropped-mask":
            cropped_mask = segmentation[x_crop_slice, y_crop_slice]
            result_tuple.append(cropped_mask)
            if dump_image:
              save_temporary_image(id(img), i, field, cropped_mask)
          elif field == "area":
            result_tuple.append(mask["area"])
          elif field == "bbox-x":
            result_tuple.append(bbox_x)
          elif field == "bbox-y":
            result_tuple.append(bbox_y)
          elif field == "bbox-w":
            result_tuple.append(bbox_w)
          elif field == "bbox-h":
            result_tuple.append(bbox_h)
          elif field == "bbox-x-center":
            result_tuple.append(bbox_x + bbox_w/2)
          elif field == "bbox-y-center":
            result_tuple.append(bbox_y + bbox_h/2)

        # Debug
        if debug:
          print(result_tuple)

        # Generate the segment
        yield (prob, tuple(result_tuple))

  # Normal segment anything model with prompting using bounding-boxes
  @scallopy.foreign_predicate(
      name=relation_decl.name.name,
      input_arg_types=arg_types[:5],
      output_arg_types=arg_types[5:],
      tag_type=float
  )
  def do_segment_with_bb(
    img: scallopy.Tensor,
    x: scallopy.u32,
    y: scallopy.u32,
    w: scallopy.u32,
    h: scallopy.u32,
  ):
    sam = config.get_sam_model()
    if sam is not None:
      predictor = SamPredictor(sam)

      cvt_img = torch.stack([img[:, :, 2], img[:, :, 1], img[:, :, 0]], dim=2)
      np_image = cvt_img.numpy()
      predictor.set_image(np_image)

      input_box = numpy.array([x, y, x + w, y + h])
      masks, iou_preds, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=None,
      )

      mask = torch.from_numpy(masks[0])
      dump_id = hash((id(img), x, y, w, h))
      prob = iou_preds[0] if not flatten_prob else 1.0

      # Get all the information for the fields
      result_tuple = [0]
      for field in output_fields:
        if field == "mask":
          result_tuple.append(mask)
          if dump_image:
            save_temporary_image(dump_id, 0, field, mask)
        elif field == "masked-cropped-image":
          raise Exception("Not suppported")
        elif field == "cropped-image":
          raise Exception("Not suppported")
        elif field == "cropped-mask":
          raise Exception("Not suppported")
        elif field == "area":
          area = area_from_mask(mask)
          result_tuple.append(area)
        elif field == "bbox-x":
          raise Exception("Not suppported")
        elif field == "bbox-y":
          raise Exception("Not suppported")
        elif field == "bbox-w":
          raise Exception("Not suppported")
        elif field == "bbox-h":
          raise Exception("Not suppported")

      # Debug
      if debug:
        print(result_tuple)

      # Generate the segment
      yield (prob, tuple(result_tuple))

  # Return the appropriate foreign predicate
  if prompt_bb:
    return do_segment_with_bb
  return do_segment


def save_temporary_image(img_id, i, kind, img_tensor):
  # First get the directory
  directory = os.path.join(DUMP_IMAGE_PATH, f"{img_id}")
  if not os.path.exists(directory):
    os.makedirs(directory)

  # Dump the image
  file_name = os.path.join(directory, f"{kind}-{i}.jpg")
  img = Image.fromarray(img_tensor.numpy())
  img.save(open(file_name, "w"))


def get_xy_cropping_slices(img_w, img_h, bb_x, bb_y, bb_w, bb_h, expand_crop_region):
  x_slice = get_cropping_slice(img_w, bb_x, bb_w, expand_crop_region)
  y_slice = get_cropping_slice(img_h, bb_y, bb_h, expand_crop_region)
  return (x_slice, y_slice)


def get_cropping_slice(size, x, w, expand_crop_region):
  start = max(0, x - expand_crop_region)
  end = min(size, x + w + expand_crop_region)
  return slice(start, end)


def area_from_mask(mask) -> int:
  return int(torch.sum(mask))
