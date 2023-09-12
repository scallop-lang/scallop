import os
import sys

from argparse import ArgumentParser

import openai

# Whether the openai plugin has been configured
CONFIGURED = False

# Number of allowed requests
NUM_ALLOWED_REQUESTS = 0

# Number of already performed requests
NUM_PERFORMED_REQUESTS = 0

# Temprature of GPT model
TEMPERATURE = 0.0

# The GPT model to use
MODEL = None

# Whether the warning has already being printed
WARNING_PRINTED = False


def setup_arg_parser(parser: ArgumentParser):
  parser.add_argument("--num-allowed-openai-request", type=int, default=100, help="Limit on the number of openai calls")
  parser.add_argument("--openai-gpt-model", type=str, default="gpt-3.5-turbo", help="The GPT model we use")
  parser.add_argument("--openai-gpt-temperature", type=float, default=0, help="The temperature for the GPT model")


def configure(args):
  global CONFIGURED
  global NUM_ALLOWED_REQUESTS
  global NUM_PERFORMED_REQUESTS
  global TEMPERATURE
  global MODEL

  # Open API
  api_key = os.getenv("OPENAI_API_KEY")
  if api_key is None:
    print("[scallop_openai] `OPENAI_API_KEY` not found, consider setting it in the environment variable", file=sys.stderr)
    return

  # Is configured
  CONFIGURED = True

  # Set the API Key
  openai.api_key = api_key

  # Set request limit
  NUM_ALLOWED_REQUESTS = args.num_allowed_openai_request
  NUM_PERFORMED_REQUESTS = 0

  # Set model
  MODEL = args.openai_gpt_model

  # Set temperature
  TEMPERATURE = args.openai_gpt_temperature


def raise_unconfigured():
  global WARNING_PRINTED
  if not WARNING_PRINTED:
    print("Open AI Plugin not configured; consider setting `OPENAI_API_KEY`", file=sys.stderr)
    WARNING_PRINTED = True
  raise Exception("Open AI Plugin not configured")


def assert_can_request():
  global CONFIGURED
  global WARNING_PRINTED
  global NUM_PERFORMED_REQUESTS
  global NUM_ALLOWED_REQUESTS

  # Check if openai API is configured
  if not CONFIGURED:
    raise_unconfigured()

  # Check openai request
  elif NUM_PERFORMED_REQUESTS > NUM_ALLOWED_REQUESTS:
    raise Exception("Exceeding allowed number of requests")

  # Ok
