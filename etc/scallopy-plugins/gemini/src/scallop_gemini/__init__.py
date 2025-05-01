from .plugin import ScallopGeminiPlugin
# import scallopy

# def setup_arg_parser(parser: ArgumentParser):
#   parser.add_argument("--key", type=str, default="12345678")

# def configure(args):
#   global MY_KEY
#   MY_KEY = args.key # load the key from cmd args

# def load_into_context(ctx: scallopy.Context):
#   # Create a foreign function that returns the loaded key
#   @scallopy.foreign_function
#   def get_key() -> str: return MY_KEY

#   # Register the `$get_key` foreign function into ctx
#   ctx.register_foreign_function(get_key)