import unittest

import scallopy
import scallop_gemini 

class TestBasics(unittest.TestCase):
  def test_ff_gemini(self):

    ctx = scallopy.Context()

    plugin_registry = scallopy.PluginRegistry(load_stdlib=True)
    plugin = scallop_gemini.ScallopGeminiPlugin()
    plugin_registry.load_plugin(plugin)
    plugin_registry.configure()
    plugin_registry.load_into_ctx(ctx)

    ctx.add_program("""
      rel questions = {
        (1, "what is the height of highest mountain in the world?"),
        (2, "are cats larger than dogs?"),
      }

      rel answer(id, $gemini(x)) = questions(id, x)

      query answer
    """)
    # ctx.add_program(""" 
    #     rel question = {
    #       (1, "Jane finished her PhD in January 5th, 2008. 2 days from today is the 10th anniversary of her PhD. What is the date 10 days ago from today?"),
    #     }
        
    #     rel answer(id, $gemini(x)) = question(id, x)
                    
    #     query answer

    # """)
    
    ctx.run()

    result = list(ctx.relation("answer"))
    print("Results:\n ")
    for i in range(len(result)):
      print(result[i])
    self.assertEqual(len(result), 2)
    self.assertEqual(result[0][0], 1)
    self.assertEqual(result[1][0], 2)
