from typing import List, Optional, Tuple, Union
import os
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

import scallopy
from scallop_gpu import get_device

FA_NAME = "owl_vit"
ERR_HEAD = f"[@{FA_NAME}]"

DUMP_IMAGE_PATH = ".tmp/scallop-owl-vit/"
HARD_OBJ_LIMIT = 1000

FIELDS = {
  "bbox-x": {
    "description": "The x-coordinate of the bounding box",
    "type": scallopy.u32,
  },
  "bbox-y": {
    "description": "The y-coordinate of the bounding box",
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
  "bbox-center-x": {
    "description": "The x coordinate of the center of the bounding box",
    "type": scallopy.u32,
  },
  "bbox-center-y": {
    "description": "The y coordinate of the center of the bounding box",
    "type": scallopy.u32,
  },
  "cropped-image": {
    "description": "The cropped image (bounded by bounding box)",
    "type": scallopy.Tensor,
  },
  "class": {
    "description": "The classified label",
    "type": scallopy.String,
  },
  "area": {
    "description": "The area of the bounding box",
    "type": scallopy.u32,
  },
}

_OWL_VIT_MODEL = None
_OWL_VIT_PROCESSOR = None


def get_owl_vit_model(checkpoint):
  global _OWL_VIT_MODEL
  global _OWL_VIT_PROCESSOR

  if _OWL_VIT_MODEL is None:
    try:
      _OWL_VIT_PROCESSOR = OwlViTProcessor.from_pretrained(checkpoint)
      _OWL_VIT_MODEL = OwlViTForObjectDetection.from_pretrained(checkpoint).to(get_device())
    except:
      return None

  return _OWL_VIT_MODEL, _OWL_VIT_PROCESSOR


@scallopy.foreign_attribute
def owl_vit(
  item,
  *,
  object_queries: Optional[List[str]] = None,
  output_fields: List[str] = ["class"],
  score_threshold: float = 0.1,
  score_multiplier: float = 1.0,
  input_obj_count: bool = False,
  expand_crop_region: int = 0,
  limit: Optional[int] = None,
  checkpoint: str = "google/owlvit-base-patch32",
  debug: bool = False,
  dump_image: bool = False,
  flatten_probability = False,
):
  # Needs to be a relation declaration, and can have only one relation
  assert item.is_relation_decl(), f"{ERR_HEAD} has to be an attribute of a relation type declaration"
  assert len(item.relation_decls()) == 1, f"{ERR_HEAD} cannot be an attribute on multiple relations]"

  # Get the argument types
  relation_decl = item.relation_decl(0)
  arg_names = [ab.name.name for ab in relation_decl.arg_bindings]
  arg_types = [ab.ty for ab in relation_decl.arg_bindings]

  # Expected input arguments
  expected_args = [(scallopy.Tensor, "image: Tensor")]

  # Object queries
  if object_queries is None:
    expected_args += [(scallopy.String, "object_queries: String")]

  # Object count
  if input_obj_count:
    expected_args += [(scallopy.u32, "obj_count: u32")]

  num_bounded = len(expected_args)

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
      if arg_name is not None:
        raise Exception(f"{ERR_HEAD} Type mismatch for `{arg_name}`. Expected `{s}`, but found arg of type `{arg_ty}`")
      else:
        raise Exception(f"{ERR_HEAD} Type mismatch for argument `{arg_id+1}`. Expected `{s}`, but found arg of type `{arg_ty}`")

  # Make sure object_queries is not empty if passed to attribute
  if object_queries:
    assert type(object_queries) == list, f"{ERR_HEAD} The parameter `object_queries` must be a list of strings"
    assert len(object_queries) > 0, f"{ERR_HEAD} The `object_queries` list must be non-empty"
    assert all([type(q) == str for q in object_queries]), f"{ERR_HEAD} The parameter `object_queries` must be a list of strings"

  # Generate foreign predicate
  @scallopy.foreign_predicate(
    name=relation_decl.name.name,
    input_arg_types=arg_types[:num_bounded],
    output_arg_types=arg_types[num_bounded:],
    tag_type=float,
  )
  def owl_vit_search(*args) -> scallopy.Facts[float, Tuple[str]]:
    # If object_queries is an arg, assume it's a semicolon-separated string
    if object_queries is None and input_obj_count:
      assert len(args) == 3
      img = args[0]
      input_object_queries = args[1].split(";")
      obj_count_limit = min(args[2], limit) if limit else args[2]
    elif object_queries is None:
      assert len(args) == 2
      img = args[0]
      input_object_queries = args[1].split(";")
      obj_count_limit = limit if limit else HARD_OBJ_LIMIT
    elif input_obj_count:
      assert len(args) == 2
      img = args[0]
      input_object_queries = object_queries
      obj_count_limit = min(args[1], limit) if limit else args[1]
    else:
      assert len(args) == 1
      img = args[0]
      input_object_queries = object_queries
      obj_count_limit = limit if limit else HARD_OBJ_LIMIT

    # return type annotation is wrong, should be Tuple[Union[str, int]] but scallop complains
    device = get_device()
    maybe_owl_vit_model = get_owl_vit_model(checkpoint)
    if maybe_owl_vit_model is None: return
    else: owl_vit_model, owl_vit_processor = maybe_owl_vit_model

    img_w, img_h, _ = img.shape
    image = Image.fromarray(img.numpy())
    inputs = owl_vit_processor(text=input_object_queries, images=image, return_tensors="pt").to(device)

    owl_vit_model.eval()
    with torch.no_grad():
      outputs = owl_vit_model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]]).to(device)

    # Convert outputs (bounding boxes and class logits) to final bounding boxes and scores
    results = owl_vit_processor.post_process_object_detection(
        outputs=outputs,
        threshold=score_threshold,
        target_sizes=target_sizes,
    )

    # Sort results by score descending
    zipped = list(zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]))
    zipped.sort(key=lambda x: x[0], reverse=True)

    # Keep track of count to stay within limit
    yielded_count = 0

    for (segment_id, (score, label, box)) in enumerate(zipped):
      if yielded_count >= obj_count_limit: break
      yielded_count += 1

      # Basic data
      final_score = score * score_multiplier if not flatten_probability else torch.ones(1)
      bbox_raw = [max(round(coord), 0) for coord in box.tolist()]
      bbox_x, bbox_y = bbox_raw[0], bbox_raw[1]
      bbox_w, bbox_h = bbox_raw[2] - bbox_x, bbox_raw[3] - bbox_y

      # Output tuple
      result_tuple = [segment_id]
      for field in output_fields:
        if field == "bbox-x":
          result_tuple.append(bbox_x)
        elif field == "bbox-y":
          result_tuple.append(bbox_y)
        elif field == "area":
          result_tuple.append(bbox_w * bbox_h)
        elif field == "bbox-w":
          result_tuple.append(bbox_w)
        elif field == "bbox-h":
          result_tuple.append(bbox_h)
        elif field == "bbox-center-x":
          result_tuple.append(int(bbox_x + bbox_w / 2))
        elif field == "bbox-center-y":
          result_tuple.append(int(bbox_y + bbox_h / 2))
        elif field == "class":
          result_tuple.append(input_object_queries[label])
        elif field == "cropped-image":
          (x_crop_slice, y_crop_slice) = get_xy_cropping_slices(img_w, img_h, bbox_y, bbox_x, bbox_h, bbox_w, expand_crop_region)
          cropped_image = img[x_crop_slice, y_crop_slice, :]
          result_tuple.append(cropped_image)
          if dump_image:
            save_temporary_image(id(img), segment_id, field, cropped_image)

      if debug:
        print((final_score, tuple(result_tuple)))

      yield (final_score, tuple(result_tuple))

  return owl_vit_search


def save_temporary_image(img_id, i, kind, img_tensor):
  from PIL import Image

  # First get the directory
  directory = os.path.join(DUMP_IMAGE_PATH, f"{img_id}")
  if not os.path.exists(directory):
    os.makedirs(directory)

  # Dump the image
  file_name = os.path.join(directory, f"{kind}-{i}.jpg")
  img = Image.fromarray(img_tensor.numpy())
  img.save(open(file_name, "w"))


def get_xy_cropping_slices(img_w, img_h, bb_x, bb_y, bb_w, bb_h, expand_crop_region=0):
  x_slice = get_cropping_slice(img_w, bb_x, bb_w, expand_crop_region)
  y_slice = get_cropping_slice(img_h, bb_y, bb_h, expand_crop_region)
  return (x_slice, y_slice)


def get_cropping_slice(size, x, w, expand_crop_region):
  start = max(0, x - expand_crop_region)
  end = min(size, x + w + expand_crop_region)
  return slice(start, end)
