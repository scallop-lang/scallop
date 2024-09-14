from argparse import ArgumentParser
from typing import Dict, List
import os
import sys

import scallopy

class ScallopGPTPlugin(scallopy.Plugin):
  def __init__(self):
    super().__init__()

    # Whether the openai plugin has been configured
    self._configured = False

    # Number of allowed requests
    self._num_allowed_requests = 100

    # Number of already performed requests
    self._num_performed_requests = 0

    # Temprature of GPT model
    self._temperature = 0.0

    # The GPT model to use
    self._default_model = "gpt-3.5-turbo"
    self._model = self._default_model

    # Whether the warning has already being printed
    self._warning_printed = False

  def setup_argparse(self, parser: ArgumentParser):
    parser.add_argument("--num-allowed-openai-request", type=int, default=100, help="Limit on the number of openai calls")
    parser.add_argument("--openai-gpt-model", type=str, default=self._default_model, help="The GPT model we use")
    parser.add_argument("--openai-gpt-temperature", type=float, default=0, help="The temperature for the GPT model")

  def configure(self, args: Dict = {}, unknown_args: List = []):
    import openai

    # Open API
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
      print("[scallop_openai] `OPENAI_API_KEY` not found, consider setting it in the environment variable", file=sys.stderr)
      return

    # Is configured
    self._configured = True

    # Set the API Key
    openai.api_key = api_key

    # Set request limit
    if "num_allowed_openai_request" in args:
      self._num_allowed_requests = args["num_allowed_openai_request"]
    self._num_performed_requests = 0

    # Set model
    if "openai_gpt_model" in args:
      self._model = args["openai_gpt_model"]

    # Set temperature
    if "openai_gpt_temperature" in args:
      self._temperature = args["openai_gpt_temperature"]

  def model(self) -> str:
    return self._model

  def temperature(self) -> float:
    return self._temperature

  def raise_unconfigured(self):
    if not self._warning_printed:
      print("Open AI Plugin not configured; consider setting `OPENAI_API_KEY`", file=sys.stderr)
      self._warning_printed = True
    raise Exception("Open AI Plugin not configured")

  def assert_can_request(self):
    # Check if openai API is configured
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
    from . import fa_gpt
    from . import ff_gpt
    from . import fp_gpt

    ctx.register_foreign_attribute(fa_extract_info.get_gpt_extract_info(self))
    ctx.register_foreign_attribute(fa_gpt.get_gpt(self))
    ctx.register_foreign_function(ff_gpt.get_gpt(self))
    ctx.register_foreign_predicate(fp_gpt.get_gpt(self))

    if scallopy.torch_tensor_enabled():
      ctx.register_foreign_attribute(fa_encoder.get_gpt_encoder(self))
