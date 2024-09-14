import openai
import scallopy

from . import ScallopGPTPlugin

def get_gpt(plugin: ScallopGPTPlugin):
  # For memoization
  STORAGE = {}

  @scallopy.foreign_function
  def gpt(prompt: str) -> str:
    if prompt in STORAGE:
      return STORAGE[prompt]
    else:
      # Make sure that we can do so
      plugin.assert_can_request()

      # Add performed requests
      plugin.increment_num_performed_request()
      response = openai.ChatCompletion.create(
        model=plugin.model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=plugin.temperature(),
      )
      choice = response["choices"][0]
      result = choice["message"]["content"].strip()

      # Store in the storage
      STORAGE[prompt] = result
      return result

  return gpt
