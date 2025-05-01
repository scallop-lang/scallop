from typing import *

import google.generativeai as genai
import torch
import os

import scallopy
from scallop_gpu import get_device

from . import ScallopGeminiPlugin

# Configure the Gemini client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
embedding_model = genai.get_model("models/embedding-001")  # Load once


FA_NAME = "gemini_encoder"
ERR_HEAD = f"[@{FA_NAME}]"

def get_gemini_encoder(plugin: ScallopGeminiPlugin):

  @scallopy.foreign_attribute
  def gemini_encoder(item, *, debug: bool = False, model: str = "models/embedding-001"):
    # Check if the annotation is on function type decl
    assert item.is_function_decl(), f"{ERR_HEAD} has to be an attribute of a function type declaration"

    # Check the input and return types
    arg_types = item.function_decl_arg_types()
    assert len(arg_types) == 1 and arg_types[0].is_string(), f"{ERR_HEAD} expects only one `String` argument"
    assert item.function_decl_ret_type().is_tensor(), f"{ERR_HEAD} expects that the return type is `Tensor`"

    STORAGE = {}

    # Generate foreign function
    @scallopy.foreign_function(name=item.function_decl_name())
    def encode_text(text: str) -> scallopy.Tensor:
      # Check memoization
      if text not in STORAGE:
        # Make sure that we can do request
        plugin.assert_can_request()

        if debug:
          print(f"{ERR_HEAD} Querying `{model}` for text `{text}`")

        # Request the embedding
        response = embedding_model.embed_content(
          content=text,
          task_type="retrieval_document",  # or "retrieval_query"
        )

        embedding = response["embedding"]

        if debug:
          print(f"{ERR_HEAD} Obtaining embedding: {embedding}")

        STORAGE[text] = embedding
        plugin.increment_num_performed_request()

      # Return tensor
      device = get_device()
      result_embedding = STORAGE[text]
      result = torch.tensor(result_embedding).to(device=device)
      return result

    return encode_text

  return gemini_encoder
