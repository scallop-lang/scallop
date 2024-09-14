from typing import *
import re
import json

import openai
import scallopy

from . import ScallopGPTPlugin

FA_NAME = "gpt"
ERR_HEAD = f"[@{FA_NAME}]"

def get_gpt(plugin: ScallopGPTPlugin):

  @scallopy.foreign_attribute
  def gpt(
    item,
    prompt: str,
    *,
    header: str = "",
    examples: List[List[str]] = [],
    model: Optional[str] = None,
    debug: bool = False,
  ):
    # Needs to be a relation declaration, and can have only one relation
    assert item.is_relation_decl(), f"{ERR_HEAD} has to be an attribute of a relation type declaration"
    assert len(item.relation_decls()) == 1, f"{ERR_HEAD} cannot be an attribute on multiple relations]"

    # Get the argument names, argument types, and adornment pattern
    relation_decl = item.relation_decl(0)
    arg_names = [ab.name.name for ab in relation_decl.arg_bindings]
    arg_types = [ab.ty for ab in relation_decl.arg_bindings]
    pattern = "".join([get_boundness(ab.adornment) for ab in relation_decl.arg_bindings])

    # Compute the number of `b`ounds and the number of `f`rees.
    regex_match = re.match("^(b*)(f+)$", pattern)
    assert regex_match is not None, f"{ERR_HEAD} pattern must start with b (optional) and ending with f (required)"
    num_bounded, num_free = len(regex_match[1]), len(regex_match[2])

    # Make sure that the types are good
    assert all([at.is_string() for at in arg_types[:num_bounded]]), f"{ERR_HEAD} annotation requires all input arguments to have `String` type"

    # The storage is special per foreign predicate
    STORAGE = {}

    # Generate the foreign predicate
    @scallopy.foreign_predicate(
      name=relation_decl.name.name,
      input_arg_types=arg_types[:num_bounded],
      output_arg_types=arg_types[num_bounded:],
      tag_type=None,
    )
    def invoke_gpt(*args):
      assert len(args) == num_bounded

      if model is None:
        local_model = plugin.model()
      else:
        local_model = model

      # Deal with STORAGE; check if the response is memoized
      storage_key = tuple(args)
      if storage_key in STORAGE:
        responses = STORAGE[storage_key]

      # Need to do a new request
      else:
        # Make sure that we can do so
        plugin.assert_can_request()

        # Fill the prompt with the inputs; replace free variables with the BLANK
        filled_prompt = fill_prompt(header, prompt, examples, args, arg_names, num_bounded)

        # Create a request to openai gpt
        plugin.increment_num_performed_request()
        system_ctx, messages = fill_template([filled_prompt])
        _, current_conversation = query_gpt_completion(system_ctx, messages, local_model)
        responses = extract_responses(current_conversation)

        # Debug print
        if debug:
          print(f"Prompt: {messages}")
          print(f"Responses: {responses}")

        # Store the response
        STORAGE[storage_key] = responses

      # Return choices
      for response in responses:
        tup = parse_choice_text(response.strip(), arg_names, arg_types, num_bounded)
        yield tup

    # Remove the item and register a foreign predicate
    return invoke_gpt

  return gpt


def get_boundness(adornment) -> str:
  return "b" if adornment and adornment.is_bound() else "f"


def fill_prompt(
  header: str,
  prompt: str,
  examples: Optional[List[Tuple[str]]],
  args: List[str],
  arg_names: List[str],
  num_bounded: int
):
  arg_patterns = ["{{" + an + "}}" for an in arg_names]
  few_shot_prompts = []
  filled_prompt = prompt

  for example in examples:
    few_shot_prompt = prompt

    few_shot_json = {}
    for (arg_pattern, fill) in zip(arg_patterns[:num_bounded], example[:num_bounded]):
      few_shot_prompt = few_shot_prompt.replace(arg_pattern, str(fill))
    for (arg_pattern, fill) in zip(arg_patterns[num_bounded:], example[num_bounded:]):
      key = arg_pattern[2:-2]
      few_shot_json[key] = str(fill)

    few_shot_prompts.append(few_shot_prompt + '\n' + json.dumps(few_shot_json))

  for (arg_pattern, fill) in zip(arg_patterns[:num_bounded], args):
    filled_prompt = filled_prompt.replace(arg_pattern, str(fill))

  # Check the prompts
  full_prompt = header
  if len(few_shot_prompts) > 0:
    full_prompt += '\n' + 'Here are a few examples \n' + '\n'.join(few_shot_prompts) + '\n'
  else:
    full_prompt += "\nPlease response in the format of JSON, where the key is the label of the blank and the value is the value to fill in\n"

  full_prompt += "Please answer the following question: \n" + filled_prompt
  return full_prompt


def parse_choice_text(text: str, arg_names: List[str], arg_types: List, num_bounded: int):
  answers_json = json.loads(text)
  answer = []
  for (arg_name, arg_type) in zip(arg_names[num_bounded:], arg_types[num_bounded:]):
    if arg_name not in answers_json:
      answer.append("")
    else:
      answer.append(arg_type.parse_value(answers_json[arg_name]))
  return tuple(answer)


def query_gpt_completion(system_ctx, messages, model):
  current_conversation = [system_ctx]
  responses = []
  for message in messages:
    current_conversation.append(message)
    response = openai.ChatCompletion.create(
      model=model,
      messages=current_conversation,
      temperature=0)
    responses.append(response)
    current_conversation.append(response['choices'][0]['message'])
  response_pairs = (responses, current_conversation)
  return response_pairs


def fill_template(texts):
  system_ctx = {"role": "system", "content": f"You are a knowledgable assistant. "}
  messages = [{"role": "user", "content": text} for text in texts]
  return system_ctx, messages


def extract_responses(conversation):
  gpt_responses = []
  for dialogue in conversation:
    if dialogue['role'] == 'assistant':
      gpt_responses.append(dialogue['content'])
  return gpt_responses
