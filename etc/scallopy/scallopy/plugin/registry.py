from typing import List, Optional, Dict, Callable
from importlib import metadata
from argparse import ArgumentParser

from .. import ScallopContext

from . import utils
from . import constants
from .plugin import Plugin
from .stdlib import cmd_arg, py_eval

class PluginRegistry:
  """
  A plugin registry
  """

  _plugins: Dict[str, Plugin]

  def __init__(
      self,
      load_stdlib: bool = False,
      load_from_entry_points: bool = False,
      read_cmd_args: bool = True,
      filter: Optional[Callable[[str], bool]] = None
  ):
    """
    Create a new plugin registry. Depending on the arguments, we either
    load the registry from entry points, or we initialize an empty
    registry.
    """
    self._plugins = dict()
    self._filter = filter
    self._read_cmd_args = read_cmd_args

    # Load plugins
    if load_stdlib:
      self.load_stdlib()
    if load_from_entry_points:
      self.load_plugins_from_entry_points()

  def load_plugin(self, plugin: Plugin, overwrite: bool = False):
    """
    Manually load a plugin

    Note: Calling this function will bypass the filter provided when
    constructing the plugin registry.

    :param plugin, a Scallop plugin
    :param overwrite, whether we want to overwrite the plugin in the registry
    """
    if plugin.name in self._plugins and not overwrite:
      raise Exception(f"Plugin {plugin.name} already registered; aborting")
    self._plugins[plugin.name] = plugin

  def load_stdlib(self):
    """
    Load all the plugins in the stdlib
    """
    self.load_plugin(cmd_arg.CmdArgPlugin())
    self.load_plugin(py_eval.PyEvalPlugin())

  def load_plugins_from_entry_points(self):
    """
    Load multiple plugins from entry points obtained by `importlib`
    """
    meta_entry_points = metadata.entry_points()
    entry_points = utils._find_entry_points(meta_entry_points, constants.SCALLOP_PLUGIN_GROUP, self._filter)
    for entry_point in entry_points:
      plugin_clazz = entry_point.load()
      plugin = plugin_clazz()
      self.load_plugin(plugin)

  def remove_all_plugins(self):
    """
    Remove all the loaded plugins in this registry
    """
    self._plugins = {}

  def loaded_plugins(self) -> List[str]:
    """
    Obtain the list of loaded plugins. Each plugin is represented
    as a string of the form `{{plugin_name}}::{{function_name}}`.
    """
    return list(self._plugins.keys())

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
    for (_, plugin) in self._plugins.items():
      plugin.setup_argparse(parser)

  def configure(self, args={}, unknown_args=[]):
    """
    Configure the plugins using the parsed arguments
    """
    # Preprocess arguments
    args = utils._process_args(args)

    # Configure the plugins
    for (_, plugin) in self._plugins.items():
      plugin.configure(args, unknown_args)

  def load_into_ctx(self, ctx: ScallopContext):
    """
    Apply the `load_into_context` function from all plugins on a newly
    created scallopy context
    """

    # Then load all the plugins
    for (_, plugin) in self._plugins.items():
      plugin.load_into_ctx(ctx)
