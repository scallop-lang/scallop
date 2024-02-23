from typing import *

from openai import OpenAI

client = OpenAI()
import torch
import pickle
import os 

import scallopy
from scallop_gpu import get_device

from . import config

FA_NAME = "gpt_encoder"
ERR_HEAD = f"[@{FA_NAME}]"

@scallopy.foreign_attribute
def gpt_encoder(item, *, debug: bool = False, model: str = "text-embedding-ada-002", storage_path: str = None):
  # Check if the annotation is on function type decl
  assert item.is_function_decl(), f"{ERR_HEAD} has to be an attribute of a function type declaration"

  # Check the input and return types
  arg_types = item.function_decl_arg_types()
  assert len(arg_types) == 1 and arg_types[0].is_string(), f"{ERR_HEAD} expects only one `String` argument"
  assert item.function_decl_ret_type().is_tensor(), f"{ERR_HEAD} expects that the return type is `Tensor`"

  if not storage_path is None and os.path.exists(storage_path):
    STORAGE = pickle.load(open(storage_path, "rb"))
  else:
    STORAGE = {}

  # Generate foreign function
  @scallopy.foreign_function(name=item.function_decl_name())
  def encode_text(text: str) -> scallopy.Tensor:
    # print("fa encoder encode text start")

    # Check memoization
    if text in STORAGE:
      # print("no need to query")
      pass
    else:
      # Make sure that we can do request
      config.assert_can_request()

      if debug:
        print(f"{ERR_HEAD} Querying `{model}` for text `{text}`")

      # Memoize the response
      response = client.embeddings.create(input=[text], model=model)
      embedding = response.data[0].embedding

      if debug:
        print(f"{ERR_HEAD} Obtaining response: {response}")

      STORAGE[text] = embedding
      if not storage_path is None:
        pickle.dump(STORAGE, open(storage_path, 'wb'))

    # Return
    device = get_device()
    result_embedding = STORAGE[text]
    result = torch.tensor(result_embedding).to(device=device)
    return result

  return encode_text
