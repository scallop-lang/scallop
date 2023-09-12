from typing import *

from . import function
from . import predicate
from . import syntax

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
  def __init__(self, msg: str):
    self.name = "error"
    self.msg = msg


class RegisterForeignFunctionAction(AttributeAction):
  def __init__(self, ff: function.ForeignFunction):
    self.name = "register_foreign_function"
    self.foreign_function = ff


class RegisterForeignPredicateAction(AttributeAction):
  def __init__(self, fp: predicate.ForeignPredicate):
    self.name = "register_foreign_predicate"
    self.foreign_predicate = fp


class ForeignAttributeProcessor:
  def __init__(self, name: str, processor: Callable):
    self.name = name
    self.processor = processor

  def process_value(self, value):
    if value.is_constant():
      return self.process_constant(value.as_constant())
    elif value.is_list():
      return [self.process_value(v) for v in value.as_list().values]
    elif value.is_tuple():
      return tuple([self.process_value(v) for v in value.as_tuple().values])
    else:
      raise Exception(f"Unknown value {value}")

  def process_constant(self, constant):
    if constant.is_string(): return constant.as_string().string
    elif constant.is_integer(): return constant.as_integer().int
    elif constant.is_boolean(): return constant.as_boolean().value
    elif constant.is_float(): return constant.as_float().float
    else: raise Exception(f"Unknown constant type {constant.key()}")

  def process_attribute(self, attr):
    pos_args = [self.process_value(arg.as_pos()) for arg in attr.args if arg.is_pos()]
    kw_args = {arg.as_kw().name.name: self.process_value(arg.as_kw().value) for arg in attr.args if arg.is_kw()}
    return (pos_args, kw_args)

  def apply(self, internal_item, internal_attr):
    item = syntax.AstNode.parse(internal_item)
    attr = syntax.AstNode.parse(internal_attr)
    (pos_args, kw_args) = self.process_attribute(attr)
    try:
      action = self.processor(item, *pos_args, **kw_args)
      return parse_processor_action(action)
    except Exception as err:
      return ErrorAction(str(err))


def parse_processor_action(action):
  if action is None:
    return NoAction()
  elif isinstance(action, AttributeAction):
    return action
  elif isinstance(action, list):
    return MultipleActions([parse_processor_action(item) for item in action])
  elif isinstance(action, function.ForeignFunction):
    return MultipleActions([RemoveItemAction(), RegisterForeignFunctionAction(action)])
  elif isinstance(action, predicate.ForeignPredicate):
    return MultipleActions([RemoveItemAction(), RegisterForeignPredicateAction(action)])
  else:
    raise Exception("Invalid return value of foreign attribute")


def foreign_attribute(func) -> ForeignAttributeProcessor:
  """
  A decorator
  """

  # Get the function name
  func_name = func.__name__

  # Get the attribute processor
  return ForeignAttributeProcessor(func_name, func)
