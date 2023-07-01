from typing import *
import re

import openai

import scallopy

from ...config import openai as openai_config

@scallopy.foreign_attribute
def gpt_complete(
  item,
  *,
  prompt: str,
  pattern: str,
  examples: List[List[str]] = [],
  debug: bool = False,
):
  # Check if the annotation is on relation type decl
  if not ("TypeDecl" in item and "Relation" in item["TypeDecl"]["node"]):
    raise Exception("`gpt` has to be an attribute of a relation type declaration")

  # Get the type decl
  relation_type_decls = item["TypeDecl"]["node"]["Relation"]["node"]["rel_types"]
  if len(relation_type_decls) > 1:
    raise Exception("`gpt` cannot be an attribute on multiple relations")
  relation_type_decl = relation_type_decls[0]["node"]

  # Get the relation name and argument types
  name = relation_type_decl["name"]["node"]["name"]
  arg_types = [(arg_type["node"]["name"]["node"]["name"], arg_type["node"]["ty"]["node"]) for arg_type in relation_type_decl["arg_types"]]

  # Get the Pattern
  regex_match = re.match("^(b*)(f+)$", pattern)
  if regex_match is None:
    raise Exception("`gpt` pattern must start with b (optional) and ending with f (required)")

  # Check if the pattern and the arg types match
  if len(arg_types) != len(pattern):
    raise Exception("`gpt` pattern must have the same length as the number of arguments")

  # Compute the number of `b`ounds and the number of `f`rees.
  num_bounded = len(regex_match[1])
  num_free = len(regex_match[2])

  # Make sure that the types are good
  if not all([ty == "String" for (_, ty) in arg_types[:num_bounded]]):
    raise Exception("`gpt` annotation requires all input arguments to have `String` type")

  # The storage is special per foreign predicate
  STORAGE = {}

  # Main function to Invoke the gpt
  def invoke_gpt(*args):
    assert len(args) == num_bounded

    # Deal with STORAGE; check if the response is memoized
    storage_key = tuple(args)
    if storage_key in STORAGE:
      response = STORAGE[storage_key]

    # Check if openai API is configured
    elif not openai_config.CONFIGURED:
      raise Exception("Open AI Plugin not configured; consider setting OPENAI_API_KEY")

    # Check openai request
    elif openai_config.NUM_PERFORMED_REQUESTS > openai_config.NUM_ALLOWED_REQUESTS:
      raise Exception("Exceeding allowed number of requests")

    # Need to do a new request
    else:
      # Fill the prompt with the inputs; replace free variables with the BLANK
      filled_prompt = fill_prompt(prompt, args, arg_types, num_bounded)

      prompt_header = ''' Answer in the format of new-line separated \"BLANK_VAR = ANSWER\".
      Here is one example:
      Father's mother is <BLANK_n>.
      BLANK_n = grandmother

      Another example is,
      The mountain Everest is of <BLANK_n> meters or <BLANK_m> feets.
      BLANK_n = 8848
      BLANK_m = 29032

      Please fill in the blank(s):
      '''

      # Create a full prompt which is "fill in the blank"
      full_prompt = f"{prompt_header}\n{filled_prompt}."

      # Create a request to openai gpt
      response = openai.Completion.create(
        model=openai_config.MODEL,
        prompt=full_prompt,
        temperature=openai_config.TEMPERATURE)

      # Debug print
      if debug:
        print(f"Prompt: {full_prompt}")
        print(f"Response: {response}")

      # Store the response
      STORAGE[storage_key] = response

    # Return choices
    for choice in response["choices"]:
      choice_text = choice["text"].strip()
      tup = parse_choice_text(choice_text, arg_types, num_bounded)
      yield tup

  # Generate the foreign predicate
  foreign_predicate = scallopy.ForeignPredicate(
    invoke_gpt,
    name,
    input_arg_types=[scallopy.predicate.Type(arg_ty) for (_, arg_ty) in arg_types[:num_bounded]],
    output_arg_types=[scallopy.predicate.Type(arg_ty) for (_, arg_ty) in arg_types[num_bounded:]],
    tag_type=None,
  )

  # Remove the item and register a foreign predicate
  return scallopy.attribute.MultipleActions([
    scallopy.attribute.RegisterForeignPredicateAction(foreign_predicate),
    scallopy.attribute.RemoveItemAction(),
  ])


def fill_prompt(prompt: str, args: List[str], arg_types: List[Tuple[str, str]], num_bounded: int):
  arg_patterns = ["{" + arg_name + "}" for (arg_name, _) in arg_types]
  free_args = ["<BLANK " + arg_name + ">" for (arg_name, _) in arg_types[num_bounded:]]
  filled_prompt = prompt
  for (arg_pattern, fill) in zip(arg_patterns[:num_bounded], args):
    filled_prompt = filled_prompt.replace(arg_pattern, str(fill))
  for (arg_pattern, fill) in zip(arg_patterns[num_bounded:], free_args):
    filled_prompt = filled_prompt.replace(arg_pattern, str(fill))
  return filled_prompt


def parse_choice_text(text: str, arg_types: List[Tuple[str, str]], num_bounded: int):
  ret_arg_regexes = [re.compile(f"{arg_name}\s*=\s*(.+)$\n?", re.MULTILINE) for (arg_name, _) in arg_types[num_bounded:]]
  matches = [next(iter(ret_arg_regex.finditer(text))) for ret_arg_regex in ret_arg_regexes]
  answers = ["" if match is None else parse_value(match[1], arg_ty) for (match, (_, arg_ty)) in zip(matches, arg_types[num_bounded:])]
  return tuple(answers)


def parse_value(text: str, arg_type: str) -> Any:
  if arg_type == "F32" or arg_type == "F64":
    return float(text)
  elif arg_type == "I8" or arg_type == "I16" or arg_type == "I32" or arg_type == "I64" or arg_type == "ISize" or \
    arg_type == "U8" or arg_type == "U16" or arg_type == "U32" or arg_type == "U64" or arg_type == "USize":
    return int(float(text))
  elif arg_type == "Bool":
    if text == "true" or text == "True": return True
    elif text == "false" or text == "False": return False
    else: return False
  elif arg_type == "String":
    return text
  else:
    raise NotImplemented()
