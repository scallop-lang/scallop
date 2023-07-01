from typing import Tuple

import openai
import scallopy

from ...config import openai as openai_config

STORAGE = {}

@scallopy.foreign_predicate
def gpt_complete(s: str) -> scallopy.Generator[None, str]:
  # Check if the storage already contains the response
  if s in STORAGE:
    response = STORAGE[s]
  else:
    if not openai_config.CONFIGURED:
      raise Exception("Open AI Plugin not configured; consider setting OPENAI_API_KEY")
    elif openai_config.NUM_PERFORMED_REQUESTS > openai_config.NUM_ALLOWED_REQUESTS:
      raise Exception("Exceeding allowed number of requests")
    else:
      # Memoize the response
      openai_config.NUM_PERFORMED_REQUESTS += 1
      response = openai.Completion.create(
        model=openai_config.MODEL,
        prompt=s,
        temperature=openai_config.TEMPERATURE)
      STORAGE[s] = response

  # Iterate through all the choices
  for choice in response["choices"]:
    result = choice["text"].strip()
    yield (result,)
