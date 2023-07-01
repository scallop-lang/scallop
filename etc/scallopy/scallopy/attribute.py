from typing import *

from . import predicate


class AttributeAction:
  def __init__(self):
    self.name = None


class MultipleActions(AttributeAction):
  def __init__(self, actions: List[AttributeAction]):
    self.name = "multiple"
    self.actions = actions


class RemoveItemAction(AttributeAction):
  def __init__(self):
    self.name = "remove_item"


class NoAction(AttributeAction):
  def __init__(self):
    self.name = "no_action"


class ErrorAction(AttributeAction):
  def __init__(self, msg):
    self.name = "error"
    self.msg = msg


class RegisterForeignPredicateAction(AttributeAction):
  def __init__(self, fp):
    self.name = "register_foreign_predicate"
    self.foreign_predicate = fp


class ForeignAttributeProcessor:
  def __init__(self, name: str, processor: Callable):
    self.name = name
    self.processor = processor

  def process_value(self, value):
    if "Constant" in value["node"]:
      return self.process_constant(value["node"]["Constant"])
    elif "List" in value["node"]:
      return [self.process_value(v) for v in value["node"]["List"]]
    else:
      raise NotImplemented()

  def process_constant(self, constant):
    if "String" in constant["node"]:
      return constant["node"]["String"]["node"]["string"]
    elif "Integer" in constant["node"]:
      return constant["node"]["Integer"]
    elif "Boolean" in constant["node"]:
      return constant["node"]["Boolean"]
    elif "Float" in constant["node"]:
      return constant["node"]["Float"]
    else:
      print(constant["node"])
      raise NotImplemented()

  def process_kw_arg_name(self, kw_arg):
    return kw_arg[0]["node"]["name"]

  def process_kw_arg_value(self, kw_arg):
    return self.process_value(kw_arg[1])

  def process_attribute(self, attr):
    # attr_name = attr["node"]["name"]["node"]["name"]
    pos_args = [self.process_value(pos_arg) for pos_arg in attr["node"]["pos_args"]]
    kw_args = {self.process_kw_arg_name(kw_arg): self.process_kw_arg_value(kw_arg) for kw_arg in attr["node"]["kw_args"]}
    return (pos_args, kw_args)

  def apply(self, item, attr):
    (pos_args, kw_args) = self.process_attribute(attr)
    try:
      result = self.processor(item, *pos_args, **kw_args)
      if result is None:
        return NoAction()
      elif isinstance(result, AttributeAction):
        return result
      else:
        raise Exception("Invalid return value of foreign attribute")
    except Exception as err:
      return ErrorAction(str(err))


def foreign_attribute(func) -> ForeignAttributeProcessor:
  """
  A decorator
  """

  # Get the function name
  func_name = func.__name__

  # Get the attribute processor
  return ForeignAttributeProcessor(func_name, func)
