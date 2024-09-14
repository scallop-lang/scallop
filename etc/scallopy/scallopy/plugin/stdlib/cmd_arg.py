from typing import Dict, List, Optional, Any
from argparse import ArgumentParser

from ... import ScallopContext
from ... import foreign_attribute, foreign_predicate

from ..plugin import Plugin

class CmdArgPlugin(Plugin):
  def __init__(self):
    super().__init__()
    self.unknown_args = []

  def configure(self, args: Dict = {}, unknown_args: List = []):
    self.unknown_args = unknown_args

  def load_into_ctx(self, ctx: ScallopContext):
    """
    Register a `@cmd_arg` foreign attribute to the Context
    """
    unknown_args = self.unknown_args

    @foreign_attribute
    def cmd_arg(item, short: str, *, long: Optional[str] = None, default: Optional[Any] = None):
      # Check if the annotation is on relation type decl
      assert item.is_relation_decl(), "[@cmd_arg] has to be an attribute of a relation type declaration"
      assert len(item.relation_decls()) == 1, "[@cmd_arg] can only be annotating one relation type declaration"
      relation_type_decl = item.relation_decl(0)
      name = relation_type_decl.name.name

      # Get the argument types
      assert len(relation_type_decl.arg_bindings) == 1, "[@cmd_arg] there can be only one argument"
      arg_type = relation_type_decl.arg_bindings[0].ty

      # Get the argument parser
      parser = ArgumentParser()
      if long is not None:
        parser.add_argument(short, long, default=default, type=arg_type.to_python_type())
      else:
        parser.add_argument(short, default=default, type=arg_type.to_python_type())

      @foreign_predicate(name=name, output_arg_types=[arg_type], tag_type=None)
      def get_arg():
        args, _ = parser.parse_known_args(unknown_args)
        if len(args.__dict__) > 0:
          key = list(args.__dict__.keys())[0]
          value = args.__dict__[key]
          if value is not None:
            yield (value,)

      # Return the foreign predicate
      return get_arg

    ctx.register_foreign_attribute(cmd_arg)
