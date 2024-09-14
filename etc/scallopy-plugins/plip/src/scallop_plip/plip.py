from typing import Dict, List, Optional, Tuple
import sys

import torch
from PIL import Image

import scallopy

from scallop_gpu import get_device

ERR_HEAD = f"[@plip]"
DELIMITER = ";"

class ScallopPLIPPlugin(scallopy.Plugin):
  def __init__(
      self,
      default_plip_model_checkpoint: str = "vinid/plip"):
    super().__init__()

    # Configurations
    self._default_plip_model_checkpoint = default_plip_model_checkpoint
    self._plip_model_checkpoint = self._default_plip_model_checkpoint
    self._plip_model = None
    self._plip_preprocess = None

  def setup_arg_parser(self, parser):
    parser.add_argument("--plip-model-checkpoint", type=str, default=self._default_plip_model_checkpoint)

  def configure(self, args: Dict = {}, unknown_args: List = []):
    if "plip_model_checkpoint" in args:
      self._plip_model_checkpoint = args["plip_model_checkpoint"]

  def get_plip_model(self, debug=False):
    if self._plip_model is None:
      try:
        if debug:
          print(f"[scallop-plip] Loading PLIP model `{self._plip_model_checkpoint}`...")
        from transformers import CLIPProcessor, CLIPModel

        model = CLIPModel.from_pretrained(self._plip_model_checkpoint)
        preprocess = CLIPProcessor.from_pretrained(self._plip_model_checkpoint)

        # model, preprocess = plip.load(_PLIP_MODEL_CHECKPOINT, device=get_device())
        self._plip_model = model
        self._plip_preprocess = preprocess

        if debug:
          print(f"[scallop-plip] Done!")
      except Exception as ex:
        if debug:
          print(ex, file=sys.stderr)
        return None

    if debug:
      print("[scallop-plip] Using loaded PLIP model")

    return (self._plip_model, self._plip_preprocess)

  def load_into_context(self, ctx: scallopy.Context):
    # Define the plip foreign attribute
    @scallopy.foreign_attribute
    def plip(
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
      def plip_classify(img: scallopy.Tensor) -> scallopy.Facts[float, Tuple[str]]:
        device = get_device()
        maybe_plip_model = self.get_plip_model(debug=debug)
        if maybe_plip_model is None:
          return

        # If successfully loaded plip, then initialize the plip models
        (plip_model, plip_preprocess) = maybe_plip_model

        # Enter non-training mode
        with torch.no_grad():
          class_prompts = [get_class_prompt(prompt, class_str) for class_str in labels]
          txt_tokens = plip.tokenize(class_prompts).to(device=device)
          img_proc = plip_preprocess(Image.fromarray(img.numpy())).unsqueeze(0).to(device=device)
          logits_per_image, _ = plip_model(img_proc, txt_tokens)
          probs = logits_per_image.softmax(dim=-1).cpu()[0]
          for prob, class_prompt, class_str in zip(probs, class_prompts, labels):
            if debug:
              print(f"[@plip_classifier] {prob:.4f} :: {class_prompt} (Class Name: {class_str})")
            if prob >= score_threshold: yield (prob, (class_str,))
            else: yield (0.0, (unknown_class,))

      # Generate the foreign predicate for dynamic labels
      @scallopy.foreign_predicate(name=relation_decl.name.name)
      def plip_classify_with_labels(img: scallopy.Tensor, list: scallopy.String) -> scallopy.Facts[float, Tuple[str]]:
        nonlocal labels
        labels = [item.strip() for item in list.split(DELIMITER)]
        return plip_classify(img)

      # Return the appropriate foreign predicate
      if labels is not None: return plip_classify
      else: return plip_classify_with_labels

    # Register the foreign attribute
    ctx.register_foreign_attribute(plip)

def get_class_prompt(prompt: Optional[str], class_name: str):
  if prompt: return prompt.replace("{{}}", class_name)
  else: return class_name
