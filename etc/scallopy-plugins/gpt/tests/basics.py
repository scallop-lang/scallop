import unittest

import scallopy
import scallop_gpt

class TestBasics(unittest.TestCase):
  def test_ff_gpt(self):

    ctx = scallopy.Context()

    plugin_registry = scallopy.PluginRegistry(load_stdlib=True)
    plugin = scallop_gpt.ScallopGPTPlugin()
    plugin_registry.load_plugin(plugin)
    plugin_registry.configure()
    plugin_registry.load_into_ctx(ctx)

    ctx.add_program("""
      rel questions = {
        (1, "what is the height of highest mountain in the world?"),
        (2, "are cats larger than dogs?"),
      }

      rel answer(id, $gpt(x)) = questions(id, x)

      query answer
    """)

    ctx.run()

    result = list(ctx.relation("answer"))
    self.assertEqual(len(result), 2)
    self.assertEqual(result[0][0], 1)
    self.assertEqual(result[1][0], 2)
