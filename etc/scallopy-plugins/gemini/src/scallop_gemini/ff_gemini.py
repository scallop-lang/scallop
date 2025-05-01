from google import genai
from google.genai import types

import scallopy
import os

from . import ScallopGeminiPlugin

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
def get_gemini(plugin: ScallopGeminiPlugin):
  # For memoization
  STORAGE = {}

  @scallopy.foreign_function
  def gemini(prompt: str) -> str:
    if prompt in STORAGE:
      return STORAGE[prompt]
    else:
      # Make sure that we can do so
      plugin.assert_can_request()

      # Add performed requests
      plugin.increment_num_performed_request()

      response = client.models.generate_content(
          model="gemini-2.0-flash",
          contents=prompt,
          config=types.GenerateContentConfig(
              temperature=0,
              top_k = 1,
              top_p = 0.5,
          ),
        )

      result = response.candidates[0].content.parts[0].text
      # print("result :", result, "\n\n\n")
      # Store in the storage
      STORAGE[prompt] = result
      return result

  return gemini