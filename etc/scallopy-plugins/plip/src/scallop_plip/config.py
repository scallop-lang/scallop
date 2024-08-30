import sys
from scallop_gpu import get_device


_DEFAULT_PLIP_MODEL_CHECKPOINT = "vinid/plip"
_PLIP_MODEL_CHECKPOINT = _DEFAULT_PLIP_MODEL_CHECKPOINT
_PLIP_MODEL = None
_PLIP_PREPROCESS = None


def setup_arg_parser(parser):
  parser.add_argument("--plip-model-checkpoint", type=str, default=_DEFAULT_PLIP_MODEL_CHECKPOINT)


def configure(args):
  global _PLIP_MODEL_CHECKPOINT
  _PLIP_MODEL_CHECKPOINT = args["plip_model_checkpoint"]


def get_plip_model(debug=False):
  global _PLIP_MODEL
  global _PLIP_PREPROCESS

  if _PLIP_MODEL is None:
    try:
      if debug:
        print(f"[scallop-plip] Loading PLIP model `{_PLIP_MODEL_CHECKPOINT}`...")
      from transformers import CLIPProcessor, CLIPModel

      model = CLIPModel.from_pretrained(_PLIP_MODEL_CHECKPOINT)
      preprocess = CLIPProcessor.from_pretrained(_PLIP_MODEL_CHECKPOINT)

      # model, preprocess = plip.load(_PLIP_MODEL_CHECKPOINT, device=get_device())
      _PLIP_MODEL = model
      _PLIP_PREPROCESS = preprocess
      
      if debug:
        print(f"[scallop-plip] Done!")
    except Exception as ex:
      if debug:
        print(ex, file=sys.stderr)
      return None

  if debug:
    print("[scallop-plip] Using loaded PLIP model")

  return (_PLIP_MODEL, _PLIP_PREPROCESS)
