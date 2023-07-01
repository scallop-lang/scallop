import openai
import os

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

def configure_openai(args):
  global CONFIGURED
  global NUM_ALLOWED_REQUESTS
  global NUM_PERFORMED_REQUESTS
  global TEMPERATURE
  global MODEL

  # Open API
  api_key = os.getenv("OPENAI_API_KEY")
  if api_key is None:
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
