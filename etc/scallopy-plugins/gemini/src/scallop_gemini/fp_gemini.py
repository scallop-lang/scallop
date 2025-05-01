from google import genai
from google.genai import types
import scallopy
import os

from . import ScallopGeminiPlugin

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
def get_gemini(plugin: ScallopGeminiPlugin):
  # A storage for @gemini foreign predicate
  STORAGE = {}

  @scallopy.foreign_predicate
  def gemini(s: str) -> scallopy.Facts[None, str]:
    # Check if the storage already contains the response
    if s in STORAGE:
      response = STORAGE[s]
    else:
      # Make sure that we can do so
      plugin.assert_can_request()

      # Memoize the response
      plugin.increment_num_performed_request()

      response = client.models.generate_content(
        model=plugin.model(),
        contents=s,
        config=types.GenerateContentConfig(
          temperature=0,
          top_k = 1,
          top_p = 0.5,
        ),
      )
      STORAGE[s] = response

    # Iterate through all the choices
    for candidate in response.candidates:
      result = candidate.content.parts[0].text.strip()
      yield (result,)

  return gemini
