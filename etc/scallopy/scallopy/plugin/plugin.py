from typing import Dict, List
from argparse import ArgumentParser

from .. import ScallopContext

class Plugin:
  """
  A Scallopy Plugin
  """
  def __init__(self, name: str = None):
    """
    Create a new Scallopy plugin.

    :param name, the name of the plugin. If the name is not provided,
                 we directly initialize it to the name of the class.
    """
    if name == None:
      class_name = type(self).__name__
      self.name = class_name
    else:
      if type(name) is not str:
        raise Exception("Expecting a string name for a Scallop plugin")
      self.name = name
    pass

  def setup_argparse(self, parser: ArgumentParser):
    """
    Setup a given argument parser.
    """
    pass

  def configure(self, args: Dict = {}, unknown_args: List =[]):
    """
    Configure the plugin with arguments and possibly unknown arguments
    parsed from command line argument.
    """
    pass

  def load_into_ctx(self, ctx: ScallopContext):
    """
    Load the plugin into a Scallop context. Should be invoking functions such
    as `ctx.register_foreign_function(...)`. Otherwise, it could make changes
    to the Scallop context by modifying it, like `ctx.add_relation(...)`.
    """
    pass
