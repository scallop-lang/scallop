from typing import List, Optional
import numpy as np
import os

from PIL import Image
import face_detection
import scallopy
from scallop_gpu import get_device

FA_NAME = "face_detection"
ERR_HEAD = f"[@{FA_NAME}]"

DUMP_IMAGE_PATH = ".tmp/scallop-face/"

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
  "cropped-image": {
    "description": "The cropped image (bounded by bounding box)",
    "type": scallopy.Tensor,
  },
}

_FACE_MODEL = None

def get_face_model():
  global _FACE_MODEL

  if _FACE_MODEL is None:
    try:
      _FACE_MODEL = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3, device=get_device())
    except:
      return None

  return _FACE_MODEL


@scallopy.foreign_attribute
def face_detection(
    item,
    output_fields: List[str],
    *,
    limit: Optional[int] = None,
    enlarge_face_factor: float = 1.5,
    debug: bool = False,
    dump_image: bool = False,
):
  # Needs to be a relation declaration, and can have only one relation
  assert item.is_relation_decl(), f"{ERR_HEAD} has to be an attribute of a relation type declaration"
  assert len(item.relation_decls()) == 1, f"{ERR_HEAD} cannot be an attribute on multiple relations]"

  # Get the argument names and argument types
  relation_decl = item.relation_decl(0)
  arg_names = [ab.name.name for ab in relation_decl.arg_bindings]
  arg_types = [ab.ty for ab in relation_decl.arg_bindings]

  # Expected input arguments
  expected_args = [(scallopy.Tensor, "image: Tensor")]

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

  # For memoization of face detection output
  STORAGE = {}

  # Generate foreign predicate
  @scallopy.foreign_predicate(
    name=relation_decl.name.name,
    input_arg_types=arg_types[:1],
    output_arg_types=arg_types[1:],
    tag_type=float,
  )
  def do_face_detect(img: scallopy.Tensor):
    img_h, img_w, _ = img.shape

    # Deal with STORAGE; check if the response is memoized
    if img in STORAGE:
      faces = STORAGE[img]
    else:
      model = get_face_model()
      if model is not None:
        faces = model.detect(np.array(img))
        STORAGE[img] = faces
      else:
        faces = []

    # Count
    yielded_count = 0

    # Iterate through all the faces
    for (i, box) in enumerate(faces):
      if debug: print(box)

      # Basic information
      x1, y1, x2, y2, prob = *map(int, box[:-1]), box[-1]
      w, h = int((enlarge_face_factor - 1) * (x2 - x1) / 2), int((enlarge_face_factor - 1) * (y2 - y1) / 2)
      x1, y1, x2, y2 = max(0, x1 - w), max(0, y1 - h), min(img_w, x2 + w), min(img_h, y2 + h)

      # Check yielded count; stop if reaching limit
      if limit is not None and yielded_count > limit: break
      else: yielded_count += 1

      # Get all the information for the fields
      result_tuple = [i]
      for field in output_fields:
        if field == "cropped-image":
          cropped_img = img[y1:y2, x1:x2, :]
          result_tuple.append(cropped_img)
          if dump_image:
            save_temporary_image(id(img), i, field, cropped_img)
        elif field == "bbox-x":
          result_tuple.append(x1)
        elif field == "bbox-y":
          result_tuple.append(y1)
        elif field == "bbox-w":
          result_tuple.append(x2 - x1)
        elif field == "bbox-h":
          result_tuple.append(y2 - y1)

      # Debug
      if debug:
        print(result_tuple)

      # Generate the face
      yield (prob, tuple(result_tuple))

  # Remove the item and register a foreign predicate
  return do_face_detect


def save_temporary_image(img_id, i, kind, img_tensor):
  # First get the directory
  directory = os.path.join(DUMP_IMAGE_PATH, f"{img_id}")
  if not os.path.exists(directory):
    os.makedirs(directory)

  # Dump the image
  file_name = os.path.join(directory, f"{kind}-{i}.jpg")
  img = Image.fromarray(img_tensor.numpy())
  img.save(open(file_name, "w"))
