from typing import List, Optional, Any
from importlib.metadata import entry_points
from argparse import ArgumentParser

import scallopy

class PluginRegistry:
  def __init__(self):
    self.setup_argparse_functions = entry_points(group="scallop.plugin.setup_arg_parser")
    self.configure_functions = entry_points(group="scallop.plugin.configure")
    self.loading_functions = entry_points(group="scallop.plugin.load_into_context")
    self.unknown_args = {}

  def loaded_plugins(self) -> List[str]:
    all_plugins = set()
    for setup_module in self.setup_argparse_functions:
      all_plugins.add(setup_module.name)
    for setup_module in self.setup_argparse_functions:
      all_plugins.add(setup_module.name)
    for setup_module in self.setup_argparse_functions:
      all_plugins.add(setup_module.name)
    return list(all_plugins)

  def dump_loaded_plugins(self):
    print("[scallopy-ext] Loaded plugins:", self.loaded_plugins())

  def setup_argument_parser(self, parser: ArgumentParser):
    for setup_module in self.setup_argparse_functions:
      setup_fn = setup_module.load()
      setup_fn(parser)

  def configure(self, args, unknown_args):
    self.unknown_args = unknown_args
    for configure_module in self.configure_functions:
      configure_fn = configure_module.load()
      configure_fn(args)

  def load_into_ctx(self, ctx: scallopy.ScallopContext):
    self.register_cmd_arg_fa(ctx)
    self.register_py_eval_fa(ctx)
    for loader_module in self.loading_functions:
      loading_fn = loader_module.load()
      loading_fn(ctx)

  def register_cmd_arg_fa(self, ctx: scallopy.ScallopContext):
    unknown_args = self.unknown_args

    @scallopy.foreign_attribute
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
      if long is not None: parser.add_argument(short, long, default=default, type=arg_type.to_python_type())
      else: parser.add_argument(short, default=default, type=arg_type.to_python_type())

      @scallopy.foreign_predicate(name=name, output_arg_types=[arg_type], tag_type=None)
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

  def register_py_eval_fa(self, ctx: scallopy.ScallopContext):
    @scallopy.foreign_attribute
    def py_eval(item, *, suppress_warning=True):
      # Check if the annotation is on function type decl
      assert item.is_function_decl(), "[@py_eval] has to be an attribute of a function type declaration"
      name, arg_types, ret_type = item.function_decl_name(), item.function_decl_arg_types(), item.function_decl_ret_type()
      assert len(arg_types) == 1 and arg_types[0].is_string(), "[@py_eval] should take only one string argument"

      # Generate the foreign function
      @scallopy.foreign_function(name=name, ret_type=ret_type, suppress_warning=suppress_warning)
      def python_evaluate(text: str): return eval(text, None, None)

      # Generate the actions
      return python_evaluate

    ctx.register_foreign_attribute(py_eval)
