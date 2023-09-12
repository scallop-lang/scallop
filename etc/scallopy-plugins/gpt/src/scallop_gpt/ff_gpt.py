import openai
import scallopy

from . import config

# For memoization
STORAGE = {}

@scallopy.foreign_function
def gpt(s: str) -> str:
  if s in STORAGE:
    return STORAGE[s]
  else:
    # Make sure that we can do so
    config.assert_can_request()

    # Add performed requests
    config.NUM_PERFORMED_REQUESTS += 1
    response = openai.ChatCompletion.create(
      model=config.MODEL,
      prompt=s,
      temperature=config.TEMPERATURE)
    choice = response["choices"][0]
    result = choice["text"].strip()
    STORAGE[s] = result
    return result
