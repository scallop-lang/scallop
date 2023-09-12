from typing import Tuple, List, Optional

import scallopy
import clip
import torch
from PIL import Image

from scallop_gpu import get_device

from .config import get_clip_model

ERR_HEAD = f"[@clip]"
DELIMITER = ";"

@scallopy.foreign_attribute
def clip(
  item,
  labels: Optional[List[str]] = None,
  *,
  prompt: Optional[str] = None,
  score_threshold: float = 0,
  unknown_class: str = "?",
  debug: bool = False,
):
  # Check if the annotation is on relation type decl
  assert item.is_relation_decl(), f"{ERR_HEAD} has to be an attribute of a relation type declaration"
  assert len(item.relation_decls()) == 1, f"{ERR_HEAD} cannot be an attribute on multiple relations"

  # Get the relation name and check argument types
  relation_decl = item.relation_decl(0)
  args = [arg for arg in relation_decl.arg_bindings]

  # Check the argument types
  assert len(args) >= 1 and args[0].ty.is_tensor() and args[0].adornment.is_bound(), f"{ERR_HEAD} first argument has to be a bounded type `Tensor`"
  if labels is not None:
    assert len(args) == 2, f"{ERR_HEAD} relation has to be of arity-2 provided the labels"
    assert args[1].ty.is_string() and (args[1].adornment is None or args[1].adornment.is_free()), f"{ERR_HEAD} second argument has to be of free type `String`"
  else:
    assert len(args) == 3, f"{ERR_HEAD} relation has to be of arity-3 given that labels need to be passed in dynamically"
    assert args[1].ty.is_string() and args[1].adornment.is_bound(), f"{ERR_HEAD} second argument has to be a bounded type `String`"
    assert args[2].ty.is_string() and (args[2].adornment is None or args[2].adornment.is_free()), f"{ERR_HEAD} third argument has to be of free type `String`"

  @scallopy.foreign_predicate(name=relation_decl.name.name)
  def clip_classify(img: scallopy.Tensor) -> scallopy.Generator[float, Tuple[str]]:
    device = get_device()
    maybe_clip_model = get_clip_model(debug=debug)
    if maybe_clip_model is None:
      return

    # If successfully loaded clip, then initialize the clip models
    (clip_model, clip_preprocess) = maybe_clip_model

    # Enter non-training mode
    with torch.no_grad():
      class_prompts = [get_class_prompt(prompt, class_str) for class_str in labels]
      txt_tokens = clip.tokenize(class_prompts).to(device=device)
      img_proc = clip_preprocess(Image.fromarray(img.numpy())).unsqueeze(0).to(device=device)
      logits_per_image, _ = clip_model(img_proc, txt_tokens)
      probs = logits_per_image.softmax(dim=-1).cpu()[0]
      for prob, class_prompt, class_str in zip(probs, class_prompts, labels):
        if debug:
          print(f"[@clip_classifier] {prob:.4f} :: {class_prompt} (Class Name: {class_str})")
        if prob >= score_threshold: yield (prob, (class_str,))
        else: yield (0.0, (unknown_class,))

  # Generate the foreign predicate for dynamic labels
  @scallopy.foreign_predicate(name=relation_decl.name.name)
  def clip_classify_with_labels(img: scallopy.Tensor, list: scallopy.String) -> scallopy.Generator[float, Tuple[str]]:
    nonlocal labels
    labels = [item.strip() for item in list.split(DELIMITER)]
    return clip_classify(img)

  # Return the appropriate foreign predicate
  if labels is not None: return clip_classify
  else: return clip_classify_with_labels


def get_class_prompt(prompt: Optional[str], class_name: str):
  if prompt: return prompt.replace("{{}}", class_name)
  else: return class_name
