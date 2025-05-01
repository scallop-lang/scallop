from argparse import ArgumentParser
from typing import Dict, List
import os
import sys

import scallopy

class ScallopGeminiPlugin(scallopy.Plugin):
  def __init__(self):
    super().__init__()

    # Whether the gemini plugin has been configured
    self._configured = False

    # Number of allowed requests
    self._num_allowed_requests = 100

    # Number of already performed requests
    self._num_performed_requests = 0

    # Temperature of gemini model
    self._temperature = 0.0

    # The Gemini model to use
    self._default_model = "gemini-2.0.flash"
    self._model = self._default_model

    # Whether the warning has already being printed
    self._warning_printed = False

  def setup_argparse(self, parser: ArgumentParser):
    parser.add_argument("--num-allowed-gemini-request", type=int, default=100, help="Limit on the number of gemini calls")
    parser.add_argument("--gemini-model", type=str, default=self._default_model, help="The gemini model we use")
    parser.add_argument("--gemini-temperature", type=float, default=0, help="The temperature for the gemini model")
    

  def configure(self, args: Dict = {}, unknown_args: List = []):
    from google import genai

    # Gemini API
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
      print("[scallop_gemini] `GEMINI_API_KEY` not found, consider setting it in the environment variable", file=sys.stderr)
      return

    # Is configured
    self._configured = True

    # Set the API Key
    genai.api_key = api_key

    # Set request limit
    if "num_allowed_gemini_request" in args:
      self._num_allowed_requests = args["num_allowed_gemini_request"]
    self._num_performed_requests = 0

    # Set model
    if "genai_model" in args:
      self._model = args["gemini_model"]

    # Set temperature
    if "genai_temperature" in args:
      self._temperature = args["gemini_temperature"]

  def model(self) -> str:
    return self._model

  def temperature(self) -> float:
    return self._temperature

  def raise_unconfigured(self):
    if not self._warning_printed:
      print("Gemini Plugin not configured; consider setting `GEMINI_API_KEY`", file=sys.stderr)
      self._warning_printed = True
    raise Exception("Gemini Plugin not configured")

  def assert_can_request(self):
    # Check if Gemini API is configured
    if not self._configured:
      self.raise_unconfigured()

    # Check openai request
    elif self._num_performed_requests > self._num_allowed_requests:
      raise Exception("Exceeding allowed number of requests")

    # Okay
    return

  def increment_num_performed_request(self):
    self._num_performed_requests += 1

  def load_into_ctx(self, ctx: scallopy.ScallopContext):
    from . import fa_encoder
    from . import fa_extract_info
    from . import ff_gemini
    from . import fp_gemini

    ctx.register_foreign_attribute(fa_extract_info.get_gemini_extract_info(self))
    ctx.register_foreign_function(ff_gemini.get_gemini(self))
    ctx.register_foreign_predicate(fp_gemini.get_gemini(self))

    if scallopy.torch_tensor_enabled():
      ctx.register_foreign_attribute(fa_encoder.get_gemini_encoder(self))
