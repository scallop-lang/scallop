import openai
import scallopy

from ...config import openai as openai_config

# For memoization
STORAGE = {}


@scallopy.foreign_function
def gpt_complete(s: str) -> str:
  if s in STORAGE:
    return STORAGE[s]
  elif not openai_config.CONFIGURED:
    raise Exception("Open AI Plugin not configured; consider setting OPENAI_API_KEY")
  elif openai_config.NUM_PERFORMED_REQUESTS > openai_config.NUM_ALLOWED_REQUESTS:
    raise Exception("Exceeding allowed number of requests")
  else:
    openai_config.NUM_PERFORMED_REQUESTS += 1
    response = openai.Completion.create(
      model=openai_config.MODEL,
      prompt=s,
      temperature=openai_config.TEMPERATURE)
    choice = response["choices"][0]
    result = choice["text"].strip()
    STORAGE[s] = result
    return result
