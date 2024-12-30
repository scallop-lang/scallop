from typing import *
import re
import json

import openai
import scallopy

from . import ScallopGPTPlugin

FA_NAME = "gpt_extract_info"
ERR_HEAD = f"[@{FA_NAME}]"

def get_gpt_extract_info(plugin: ScallopGPTPlugin):
  _RESPONSE_STORAGE = {}
  SYS_PROMPT = "You are a helpful assistant and diligent programmer. Please answer everything in the JSON format. "
  COT_PROMPT = "Please give your reasoning step by step before giving the JSON output."

  @scallopy.foreign_attribute
  def gpt_extract_info(
    item,
    *,
    header: str = "",
    prompts: List[str],
    examples: List[Tuple[List[str], List[List[Tuple[str, ...]]]]] = [],
    model: str = None,
    cot: List[bool] = None,
    error_retry_limit: int = 1,
    error_retry_message: str = "Sorry, the answer is invalid. We expect you to generate a JSON list.",
    debug: bool = False,
  ):
    # Check if the annotation is on relation type decl
    assert item.is_relation_decl(), f"{ERR_HEAD} has to be an attribute of a relation type declaration"
    relation_decl_infos = [get_rel_decl_info(r) for r in item.relation_decls()]

    # Make sure that all the relation type declarations have the same bounded arguments (name and type)
    (_, fst_arg_names, fst_arg_types, _, fst_num_bounded) = relation_decl_infos[0]
    input_arg_names = fst_arg_names[:fst_num_bounded]
    input_arg_types = fst_arg_types[:fst_num_bounded]
    for (i, at) in enumerate(input_arg_types):
      assert at.is_string(), f"{ERR_HEAD} The `{i+1}`th input variable has to be string, found `{at}`"
    for (_, curr_arg_names, curr_arg_types, _, curr_num_bounded) in relation_decl_infos[1:]:
      assert curr_num_bounded == fst_num_bounded, f"{ERR_HEAD} The number of input variables are differing"
      for j in range(fst_num_bounded):
        an, at = curr_arg_names[j], curr_arg_types[j]
        assert an == input_arg_names[j], f"{ERR_HEAD} The `{j+1}`th input variables need to have the same name, found `{an}`"
        assert at.is_string(), f"{ERR_HEAD} The `{j+1}`th input variable has to be a string, found `{at}`"

    # Make sure that every relation declaration gets a prompt
    assert len(prompts) == len(relation_decl_infos), f"{ERR_HEAD} The number of prompts must match the number of relations"

    # Make sure examples are formatted correctly
    for (bound, output) in examples:
      assert len(bound) == fst_num_bounded, f"{ERR_HEAD} The number of input variables in each example must match the number of input variables in the relations"
      assert len(output) == len(relation_decl_infos), f"{ERR_HEAD} The number of output relations in each example must match the number of declared relations"
      for i in range(len(output)):
        for fact in output[i]:
          assert len(fact) == len(relation_decl_infos[i][1]) - fst_num_bounded, f"{ERR_HEAD} The arity of the fact tuples in each example must match the arity of their corresponding declared relation"

    # Make sure that every relation declaration gets a cot
    assert cot is None or len(cot) == len(relation_decl_infos), f"{ERR_HEAD} The length of `cot` must match the number of relations"

    # Get the gpt invoker
    gpt_invoker = generate_gpt_invoker(relation_decl_infos, input_arg_names, header, prompts, examples, model, cot, debug)

    # Get the foreign predicates
    fps = [generate_foreign_predicate(gpt_invoker, relation_decl_infos[i]) for i in range(len(relation_decl_infos))]
    return fps

  def generate_foreign_predicate(
    gpt_invoker: Callable,
    relation_type
  ):
    # Get the relation name and argument types
    (name, arg_names, arg_types, _, num_bounded) = relation_type

    # Get the foreign predicate for invoking gpt
    @scallopy.foreign_predicate(
      name,
      input_arg_types=arg_types[:num_bounded],
      output_arg_types=arg_types[num_bounded:],
      tag_type=None)
    def invoke_gpt_and_get_result(*args):
      response = gpt_invoker(*args)
      if response is not None:
        if name in response:
          if type(response[name]) == list:
            for response_json in response[name]:
              response_tuple = []
              for (arg_name, arg_type) in zip(arg_names[num_bounded:], arg_types[num_bounded:]):
                if arg_name in response_json:
                  response_tuple.append(arg_type.parse_value(response_json[arg_name]))
                else:
                  continue
              yield tuple(response_tuple)

    return invoke_gpt_and_get_result


  def generate_gpt_invoker(
    relation_type_infos: List,
    input_arg_names: List[str],
    header: str,
    prompts: List[str],
    examples: List[Tuple[List[str], List[List[Tuple[str, ...]]]]],
    model: Optional[str] = None,
    cot: List[bool] = None,
    debug: bool = False,
  ):
    def format_input(args):
      return ''.join([f"{arg_name}: {arg_value}\n" for (arg_name, arg_value) in zip(input_arg_names, args)])

    def invoke_gpt(*args):
      # Get the model to use
      if model is None: local_model = plugin.model()
      else: local_model = model

      # Turn cot on or off
      if cot is None: cot_arr = [False] * len(relation_type_infos)
      else: cot_arr = cot

      # Check memoization
      memoization_key = tuple(args)
      if memoization_key in _RESPONSE_STORAGE:
        return _RESPONSE_STORAGE[memoization_key]

      # Check whether we can call GPT
      plugin.assert_can_request()
      plugin.increment_num_performed_request()

      # The result to return
      result = {}

      # Debug
      if debug: print(f"Processing question: {memoization_key}")

      # Current messages
      current_messages = [
        {"role": "system", "content": f"{SYS_PROMPT} {header}"},
      ]

      # Add examples
      for (bound, output) in examples:
        for (idx, ((_, arg_names, _, _, _), prompt, output_arg_values)) in enumerate(zip(relation_type_infos, prompts, output)):
          call_header = format_input(bound) if idx == 0 else ""
          current_messages.append({"role": "user", "content": f"{call_header}{prompt}"})
          relation = []
          for arg_values in output_arg_values:
            output_json = {}
            for (arg_name, arg_value) in zip(arg_names[len(args):], arg_values):
              output_json[arg_name] = arg_value
            relation.append(output_json)
          current_messages.append({"role": "assistant", "content": json.dumps(relation)})

      # Make the subsequent calls to GPT
      for (idx, (relation_type_info, prompt)) in enumerate(zip(relation_type_infos, prompts)):
        (name, arg_names, _, _, num_bounded) = relation_type_info
        call_header = format_input(args) if idx == 0 else ""
        cot_response_content = ""
        if cot_arr[idx]:
          cot_message = {"role": "user", "content": f"{call_header}{prompt} {COT_PROMPT}"}
          cot_response = openai.ChatCompletion.create(
            model=local_model,
            messages=current_messages + [cot_message],
            temperature=plugin.temperature())
          cot_response_content = cot_response["choices"][0]["message"]["content"]
        current_message = {"role": "user", "content": f"{call_header}{cot_response_content} {prompt}"}
        if debug: print(f"> Sending request: {current_message}")
        current_messages.append(current_message)
        curr_response = openai.ChatCompletion.create(
          model=local_model,
          messages=current_messages,
          temperature=plugin.temperature(),
          response_format={"type": "json_object"})
        curr_response_message = curr_response["choices"][0]["message"]
        if debug: print(f"> Obtained response: {curr_response_message}")
        current_messages.append(curr_response_message)
        curr_response_json = parse_response_json(curr_response_message, arg_names[num_bounded:])
        result[name] = curr_response_json

      # Do the memoization and return
      _RESPONSE_STORAGE[memoization_key] = result
      return result

    return invoke_gpt



  def get_rel_decl_info(rel_decl):
    arg_names = [ab.name.name for ab in rel_decl.arg_bindings]
    arg_types = [ab.ty for ab in rel_decl.arg_bindings]
    pattern = "".join([get_boundness(ab.adornment) for ab in rel_decl.arg_bindings])

    # Get the pattern
    regex_match = re.match("^(b*)(f+)$", pattern)
    assert regex_match is not None, f"{ERR_HEAD} pattern must start with b (optional) and ending with f (required)"

    # Compute the number of `b`ounds and the number of `f`rees.
    num_bounded = len(regex_match[1])

    return (rel_decl.name.name, arg_names, arg_types, pattern, num_bounded)


  def get_boundness(adornment) -> str:
    return "b" if adornment and adornment.is_bound() else "f"


  def parse_response_json(response, expected_keys):
    role = response["role"]
    if role == "assistant":
      response_json = json.loads(response["content"])
      if type(response_json) == list:
        return response_json
      elif type(response_json) == dict:
        if len(response_json) == 1:
          key = list(response_json.keys())[0]
          if len(expected_keys) == 1 and key == expected_keys[0]:
            return [response_json]
          elif type(response_json[key]) == list:
            return response_json[key]
          elif type(response_json[key]) == dict:
            return [response_json[key]]
        return [response_json]
      else:
        raise Exception(f"{ERR_HEAD} Unexpected output json type {type(response_json)}")
    else:
      raise Exception(f"{ERR_HEAD} Unknown GPT response role: {role}")

  return gpt_extract_info
