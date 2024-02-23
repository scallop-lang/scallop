from typing import Tuple

import openai
import scallopy

from . import config

STORAGE = {}

@scallopy.foreign_predicate
def gpt(s: str) -> scallopy.Facts[None, str]:
  # Check if the storage already contains the response
  if s in STORAGE:
    response = STORAGE[s]
  else:
    # Make sure that we can do so
    config.assert_can_request()

    # Memoize the response
    config.NUM_PERFORMED_REQUESTS += 1
    response = openai.ChatCompletion.create(
      model=config.MODEL,
      prompt=s,
      temperature=config.TEMPERATURE)
    STORAGE[s] = response

  # Iterate through all the choices
  for choice in response["choices"]:
    result = choice["text"].strip()
    yield (result,)
