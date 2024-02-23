import unittest

import scallopy
import scallopy_ext

class TestConfigure(unittest.TestCase):
  def test_loading_no_plugin(self):
    plugin_registry = scallopy_ext.PluginRegistry(load_from_entry_points=False)
    plugins = plugin_registry.loaded_plugins()
    assert len(plugins) == 0, "there should be no loaded plugins"

  def test_load_a_plugin(self):
    def configure(args): pass
    plugin_registry = scallopy_ext.PluginRegistry(load_from_entry_points=False)
    plugins = plugin_registry.loaded_plugins()
    assert len(plugins) == 0, "there should be no loaded plugins"
    plugin_registry.load_plugin("temp", configure=configure)
    plugins = plugin_registry.loaded_plugins()
    assert plugins == ["temp::configure"], "there should one configure plugin"
