from typing import *

import scallopy
from scallop_gpu import get_device


FA_NAME = "roberta_encoder"
ERR_HEAD = f"[@{FA_NAME}]"

_MODELS = {}


def get_roberta_model(checkpoint):
  global _MODELS
  if checkpoint in _MODELS:
    (tokenizer, model) = _MODELS[checkpoint]
  else:
    from transformers import RobertaTokenizer, RobertaModel
    device = get_device()
    tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
    model = RobertaModel.from_pretrained(checkpoint).to(device)
    _MODELS[checkpoint] = (tokenizer, model)
  return (tokenizer, model)


@scallopy.foreign_attribute
def roberta_encoder(
  item,
  *,
  checkpoint: str = "roberta-base",
):
  # Check if the annotation is on function type decl
  assert item.is_function_decl(), f"{ERR_HEAD} has to be an attribute of a function type declaration"

  # Check the input and return types
  arg_types = item.function_decl_arg_types()
  assert len(arg_types) == 1 and arg_types[0].is_string(), f"{ERR_HEAD} expects only one `String` argument"
  assert item.function_decl_ret_type().is_tensor(), f"{ERR_HEAD} expects that the return type is `Tensor`"

  # Generate foreign function
  @scallopy.foreign_function(name=item.function_decl_name())
  def encode_text(text: str) -> scallopy.Tensor:
    (tokenizer, model) = get_roberta_model(checkpoint)
    device = get_device()

    # prepare input
    encoded_input = tokenizer(text, return_tensors='pt').to(device)

    # forward pass
    output = model(**encoded_input)
    output = output.last_hidden_state[0, -1, :]

    return output

  return encode_text
