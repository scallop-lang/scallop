from ... import ScallopContext
from ... import foreign_attribute, foreign_function

from ..plugin import Plugin

class PyEvalPlugin(Plugin):
  def __init__(self):
    super().__init__()

  def load_into_ctx(self, ctx: ScallopContext):
    """
    Register a `@py_eval` foreign attribute to the Context
    """

    @foreign_attribute
    def py_eval(item, *, suppress_warning=True):
      # Check if the annotation is on function type decl
      assert item.is_function_decl(), "[@py_eval] has to be an attribute of a function type declaration"
      name, arg_types, ret_type = item.function_decl_name(), item.function_decl_arg_types(), item.function_decl_ret_type()
      assert len(arg_types) == 1 and arg_types[0].is_string(), "[@py_eval] should take only one string argument"

      # Generate the foreign function
      @foreign_function(name=name, ret_type=ret_type, suppress_warning=suppress_warning)
      def python_evaluate(text: str): return eval(text, None, None)

      # Generate the actions
      return python_evaluate

    ctx.register_foreign_attribute(py_eval)
