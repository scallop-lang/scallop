from typing import *
import re
import json
from pydantic import ConfigDict
from google import genai
from google.genai import types
import scallopy
import os

from . import ScallopGeminiPlugin

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
FA_NAME = "gemini_extract_info"
ERR_HEAD = f"[@{FA_NAME}]"


def get_gemini_extract_info(plugin: ScallopGeminiPlugin):
  _RESPONSE_STORAGE = {}
  SYS_PROMPT = "You are a helpful assistant and diligent programmer. Please answer everything in JSON format."
  COT_PROMPT = "Please give your reasoning step by step before giving the output."

  @scallopy.foreign_attribute
  def gemini_extract_info(
    item,
    *,
    header: str = "",
    prompts: List[str],
    examples: List[Tuple[List[str], List[List[Tuple[str, ...]]]]] = [],
    model: str = "gemini-2.0-flash",
    cot: List[bool] = None,
    error_retry_limit: int = 1,
    error_retry_message: str = "Sorry, the answer is invalid. We expect you to generate a JSON list.",
    debug: bool = False,
  ):
    # Check if the annotation is on relation type decl
    assert item.is_relation_decl(), f"{ERR_HEAD} has to be an attribute of a relation type declaration"
    relation_decl_infos = [get_rel_decl_info(r) for r in item.relation_decls()]
    
    # # Make sure that all the relation type declarations have the same bounded arguments (name and type)
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

    # # Make sure that every relation declaration gets a prompt
    assert len(prompts) == len(relation_decl_infos), f"{ERR_HEAD} The number of prompts must match the number of relations"
    
    # # Make sure examples are formatted correctly
    for (bound, output) in examples:
      assert len(bound) == fst_num_bounded, f"{ERR_HEAD} The number of input variables in each example must match the number of input variables in the relations"
      assert len(output) == len(relation_decl_infos), f"{ERR_HEAD} The number of output relations in each example must match the number of declared relations"
      for i in range(len(output)):
        for fact in output[i]:
          assert len(fact) == len(relation_decl_infos[i][1]) - fst_num_bounded, f"{ERR_HEAD} The arity of the fact tuples in each example must match the arity of their corresponding declared relation"

    # # Make sure that every relation declaration gets a cot
    assert cot is None or len(cot) == len(relation_decl_infos), f"{ERR_HEAD} The length of `cot` must match the number of relations"

    # # Get the gemini invoker
    gemini_invoker = generate_gemini_invoker(relation_decl_infos, input_arg_names, header, prompts, examples, model, cot, debug)

    # # Get the foreign predicates
    fps = [generate_foreign_predicate(gemini_invoker, relation_decl_infos[i]) for i in range(len(relation_decl_infos))]
    return fps

  def generate_foreign_predicate(
    gemini_invoker: Callable,
    relation_type
  ):
    # Get the relation name and argument types
    (name, arg_names, arg_types, _, num_bounded) = relation_type
    
    # Get the foreign predicate for invoking gemini
    @scallopy.foreign_predicate(
      name,
      input_arg_types=arg_types[:num_bounded],
      output_arg_types=arg_types[num_bounded:],
      tag_type=None)
    def invoke_gemini_and_get_result(*args):
      response = gemini_invoker(*args)
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

    return invoke_gemini_and_get_result


  def generate_gemini_invoker(
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
    
    def invoke_gemini(*args):
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

      # Check whether we can call gemini
      plugin.assert_can_request()
      plugin.increment_num_performed_request()

      # The result to return
      result = {}

      # Debug
      if debug: print(f"Processing question: {memoization_key}")

      # Current messages
      current_messages = SYS_PROMPT + header + "\n"

      # Add examples
      for (bound, output) in examples:
        for (idx, ((_, arg_names, _, _, _), prompt, output_arg_values)) in enumerate(zip(relation_type_infos, prompts, output)):
          call_header = format_input(bound) if idx == 0 else ""
          current_messages += call_header + prompt + " "

          for arg_values in output_arg_values:
            output = ""
            for (arg_name, arg_value) in zip(arg_names[len(args):], arg_values):
              output += arg_name + ": " + arg_value + " "
          current_messages += output + "\n"
        current_messages += "\n"

      # Make the subsequent calls to Gemini
      for (idx, (relation_type_info, prompt)) in enumerate(zip(relation_type_infos, prompts)):
        (name, arg_names, _, _, num_bounded) = relation_type_info
        call_header = format_input(args) if idx == 0 else ""
        cot_response_content = ""
        if cot_arr[idx]:
          cot_message = call_header + prompt + COT_PROMPT
          cot_response = client.models.generate_content(
            model=local_model,
            contents=current_messages + cot_message,
            config=types.GenerateContentConfig(
              temperature=0,
              top_k = 1,
              top_p = 0.5,
            ),
          )


          cot_response_content = cot_response.candidates[0].content.parts[0].text
        current_message = call_header + cot_response_content + prompt
        
        if debug: print(f"> Sending request: {current_message}")
        current_messages += current_message
        
        curr_response = client.models.generate_content(
          model=local_model,
          contents=current_messages,
          config=types.GenerateContentConfig(
              temperature=0,
              top_k = 1,
              top_p = 0.5,
          ),
        )
        curr_response_message = curr_response.candidates[0].content.parts[0].text

        if debug: print(f"> Obtained response: {curr_response_message}\n")
        current_messages += curr_response_message
        
        json_strings = [block.strip() for block in curr_response_message.strip().split('```') if block.strip()]
        response_dict = {}
        for json_str in json_strings: 
          try: # Remove "json" if it exists at the start of the string 
              if json_str.startswith("json"): json_str = json_str[4:].strip() # Remove the 'json' and any leading spaces
              # Parse the cleaned JSON string
              parsed_json = json.loads(json_str)

              # Update the dictionary with keys and their corresponding values
              response_dict.update(parsed_json)
              
          except json.JSONDecodeError:
              pass
        pred_args = arg_names[num_bounded:]

        filtered_json = extract_key_groups(response_dict, pred_args)

        result[name] = filtered_json
        print(filtered_json, "\n")
      # Do the memoization and return
      _RESPONSE_STORAGE[memoization_key] = result
      return result

    return invoke_gemini

  
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


  def extract_key_groups(data, target_keys):
    extracted_list = []
    print(data)
    def search(data):
        if isinstance(data, dict):
            # Check if all target keys are directly in this dict
            if all(key in data for key in target_keys):
                entry = {}
                for key in target_keys:
                    value = data[key]
                    # Unwrap if the value is a dict with the target keys again
                    if isinstance(value, dict) and all(k in value for k in target_keys):
                        entry.update(value)
                    else:
                        entry[key] = value
                extracted_list.append(entry)
            else:
                # Continue searching in nested structures
                for value in data.values():
                    search(value)
        elif isinstance(data, list):
            for item in data:
                search(item)

    search(data)
    return extracted_list

  def get_boundness(adornment) -> str:
    return "b" if adornment and adornment.is_bound() else "f"


  return gemini_extract_info
