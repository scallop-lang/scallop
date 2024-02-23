from typing import List, Optional, Any, Callable
from importlib import metadata
from argparse import ArgumentParser

import scallopy

from . import utils
from . import constants

class PluginRegistry:
  setup_argparse_functions: List[utils.EntryPoint]
  configure_functions: List[utils.EntryPoint]
  loading_functions: List[utils.EntryPoint]

  def __init__(
      self,
      load_from_entry_points: bool = True,
      filter: Optional[Callable[[str], bool]] = None
  ):
    """
    Create a new plugin registry. Depending on the arguments, we either
    load the registry from entry points, or we initialize an empty
    registry.
    """
    self.setup_argparse_functions = []
    self.configure_functions = []
    self.loading_functions = []
    self.filter = filter

    # Load plugins
    if load_from_entry_points: self.load_plugins_from_entry_points()

    # The unknown arguments is set to empty by default
    self.unknown_args = []

  def load_plugin(
      self,
      name: str,
      setup_arg_parser: Optional[Callable] = None,
      configure: Optional[Callable] = None,
      load_into_context: Optional[Callable] = None,
  ):
    """
    Manually load a plugin, given its `name`, and `setup_arg_parser`,
    `configure`, `load_into_context` functions.

    Note: Calling this function will bypass the filter provided when
    constructing the plugin registry.
    """
    if setup_arg_parser is not None:
      self.setup_argparse_functions.append(utils.CustomEntryPoint(name, constants.SETUP_ARG_PARSER, setup_arg_parser))
    if configure is not None:
      self.configure_functions.append(utils.CustomEntryPoint(name, constants.CONFIGURE, configure))
    if load_into_context is not None:
      self.loading_functions.append(utils.CustomEntryPoint(name, constants.LOAD_INTO_CONTEXT, load_into_context))

  def load_plugins_from_entry_points(self):
    """
    Load multiple plugins from entry points obtained by `importlib`
    """
    eps = metadata.entry_points()
    utils._dedup_extend(self.setup_argparse_functions, utils._find_entry_points(eps, constants.SETUP_ARG_PARSER_GROUP, self.filter))
    utils._dedup_extend(self.configure_functions, utils._find_entry_points(eps, constants.CONFIGURE_GROUP, self.filter))
    utils._dedup_extend(self.loading_functions, utils._find_entry_points(eps, constants.LOAD_INTO_CONTEXT_GROUP, self.filter))

  def remove_all_plugins(self):
    """
    Remove all the loaded plugins in this registry
    """
    self.setup_argparse_functions = []
    self.configure_functions = []
    self.loading_functions = []

  def loaded_plugins(self) -> List[str]:
    """
    Obtain the list of loaded plugins. Each plugin is represented
    as a string of the form `{{plugin_name}}::{{function_name}}`.
    """
    all_plugins = set()
    for setup_module in self.setup_argparse_functions:
      all_plugins.add(f"{setup_module.name}::{constants.SETUP_ARG_PARSER}")
    for config_module in self.configure_functions:
      all_plugins.add(f"{config_module.name}::{constants.CONFIGURE}")
    for loading_module in self.loading_functions:
      all_plugins.add(f"{loading_module.name}::{constants.LOAD_INTO_CONTEXT}")
    return list(all_plugins)

  def dump_loaded_plugins(self):
    """
    Print the list of loaded plugins
    """
    print("[scallopy-ext] Loaded plugins:", ", ".join(self.loaded_plugins()))

  def setup_argument_parser(self, parser: ArgumentParser):
    """
    Apply the `setup_arg_parser` function from all plugins on a new
    argument parser.
    """
    for setup_module in self.setup_argparse_functions:
      setup_fn = setup_module.load()
      setup_fn(parser)

  def configure(self, args={}, unknown_args=[]):
    """
    Configure the plugins using the parsed arguments
    """
    # Preprocess arguments
    args = utils._process_args(args)

    # Configure the plugins
    self.unknown_args = unknown_args
    for configure_module in self.configure_functions:
      configure_fn = configure_module.load()
      configure_fn(args)

  def load_into_ctx(self, ctx: scallopy.Context):
    """
    Apply the `load_into_context` function from all plugins on a newly
    created scallopy context
    """

    # First register two built-in attributes
    self.register_cmd_arg_fa(ctx)
    self.register_py_eval_fa(ctx)

    # Then load all the plugins
    for loader_module in self.loading_functions:
      loading_fn = loader_module.load()
      loading_fn(ctx)

  def register_cmd_arg_fa(self, ctx: scallopy.Context):
    """
    Register a `@cmd_arg` foreign attribute to the Context
    """
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

  def register_py_eval_fa(self, ctx: scallopy.Context):
    """
    Register a `@py_eval` foreign attribute to the Context
    """

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
