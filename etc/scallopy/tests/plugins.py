import unittest

import scallopy

class TestConfigure(unittest.TestCase):
  def test_loading_no_plugin(self):
    # Create a plugin registry
    plugin_registry = scallopy.PluginRegistry()
    plugins = plugin_registry.loaded_plugins()
    assert len(plugins) == 0, "there should be no loaded plugins"

  def test_load_a_plugin(self):
    # When initializing a plugin registry, there is no plugin
    plugin_registry = scallopy.PluginRegistry()
    plugins = plugin_registry.loaded_plugins()
    assert len(plugins) == 0, "there should be no loaded plugins"

    # After inserting a plugin, there should be one plugin
    class Temp(scallopy.Plugin): pass
    plugin_registry.load_plugin(Temp())
    plugins = plugin_registry.loaded_plugins()
    assert plugins == ["Temp"], "there should one configure plugin"

  def test_py_eval(self):
    # When initializing a plugin registry, there is no plugin
    plugin_registry = scallopy.PluginRegistry(load_stdlib=True)
    plugins = plugin_registry.loaded_plugins()
    assert "PyEvalPlugin" in plugins, "there should be a PyEvalPlugin in plugin registry"

    # We want to create a context with plugins loaded
    ctx = scallopy.Context()
    plugin_registry.load_into_ctx(ctx)

    # Adding a program that uses py_eval and run it
    ctx.add_program("""
      @py_eval
      type $eval_bool(input: String) -> bool
      rel my_num = {3, 5, 8, 10}
      rel is_odd(x, $eval_bool($format("{} % 2 == 1", x))) = my_num(x)
    """)
    ctx.run()
    result = list(ctx.relation("is_odd"))
    self.assertListEqual(result, [(3, True), (5, True), (8, False), (10, False)])

  def test_cmd_args(self):
    # When initializing a plugin registry, there is no plugin
    plugin_registry = scallopy.PluginRegistry(load_stdlib=True)
    plugins = plugin_registry.loaded_plugins()
    assert "CmdArgPlugin" in plugins, "there should be a CmdArgPlugin in plugin registry"

    # We want to create a context with plugins loaded
    ctx = scallopy.Context()
    plugin_registry.configure({}, ["--key", "some-key"])
    plugin_registry.load_into_ctx(ctx)

    # Adding a program that uses cmd_arg and run it
    ctx.add_program("""
      @cmd_arg("--key")
      type key(s: String)
      rel result(s) = key(s)
    """)
    ctx.run()
    result = list(ctx.relation("result"))
    self.assertListEqual(result, [("some-key",)])

if __name__ == '__main__':
  unittest.main()
