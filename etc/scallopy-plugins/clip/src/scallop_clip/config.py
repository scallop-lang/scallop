import sys
from scallop_gpu import get_device


_DEFAULT_CLIP_MODEL_CHECKPOINT = "ViT-B/32"
_CLIP_MODEL_CHECKPOINT = _DEFAULT_CLIP_MODEL_CHECKPOINT
_CLIP_MODEL = None
_CLIP_PREPROCESS = None


def setup_arg_parser(parser):
  parser.add_argument("--clip-model-checkpoint", type=str, default=_DEFAULT_CLIP_MODEL_CHECKPOINT)


def configure(args):
  global _CLIP_MODEL_CHECKPOINT
  _CLIP_MODEL_CHECKPOINT = args.clip_model_checkpoint


def get_clip_model(debug=False):
  global _CLIP_MODEL
  global _CLIP_PREPROCESS

  if _CLIP_MODEL is None:
    try:
      if debug:
        print(f"[scallop-clip] Loading CLIP model `{_CLIP_MODEL_CHECKPOINT}`...")
      import clip
      model, preprocess = clip.load(_CLIP_MODEL_CHECKPOINT, device=get_device())
      _CLIP_MODEL = model
      _CLIP_PREPROCESS = preprocess
      if debug:
        print(f"[scallop-clip] Done!")
    except Exception as ex:
      if debug:
        print(ex, file=sys.stderr)
      return None

  if debug:
    print("[scallop-clip] Using loaded CLIP model")

  return (_CLIP_MODEL, _CLIP_PREPROCESS)
