import sys

# Argument parser
import argparse

# Prompting for REPL
import prompt_toolkit
from tabulate import tabulate

# Scallop modules
import scallopy


TABLE_FMT = "rounded_outline"
MULTILINE_TABLE_FMT = "fancy_grid"
MAX_COL_WIDTH = 120


def argument_parser(plugin_registry: scallopy.PluginRegistry):
  parser = argparse.ArgumentParser("scallop", description="Scallop language command line interface")
  parser.add_argument("file", nargs="?", default=None, help="The file to execute")
  parser.add_argument("-p", "--provenance", type=str, default="unit", help="The provenance to pick")
  parser.add_argument("-k", "--top-k", default=3, type=int, help="The `k` to use when applying `top-k` related provenance")
  parser.add_argument("--wmc-with-disjunctions", action="store_true", help="Whether to use disjunctions when performing weighted model counting (WMC)")
  parser.add_argument("-m", "--module", type=str, default=None, help="Load module in interactive mode")
  parser.add_argument("--iter-limit", type=int, default=100, help="Iteration limit")
  parser.add_argument("--debug-front", action="store_true", help="Dump Front IR")
  parser.add_argument("--debug-back", action="store_true", help="Dump Back IR")
  parser.add_argument("--debug-ram", action="store_true", help="Dump RAM IR")
  parser.add_argument("--dump-loaded-plugins", action="store_true", help="Dump loaded scallopy plugins")
  parser.add_argument("--no-plugin", action="store_true", help="Do not load any plugin")

  # Get the basic arguments
  base_args, _ = parser.parse_known_args()

  # Setup using plugin registry
  if not base_args.no_plugin:
    plugin_registry.load_stdlib()
    plugin_registry.load_plugins_from_entry_points()
    plugin_registry.setup_argument_parser(parser)

  # Return the final parser
  return parser


def cmd_args(plugin_registry: scallopy.PluginRegistry):
  parser = argument_parser(plugin_registry)
  return parser.parse_known_args()


def print_relation(ctx: scallopy.ScallopContext, relation_name: str):
  # First get the relation
  if ctx.is_probabilistic():
    relation = [(f"{prob:.4f}", *tup) for (prob, tup) in ctx.relation(relation_name)]
  else:
    relation = list(ctx.relation(relation_name))

  # Check if relation is empty
  if len(relation) == 0:
    print("{}")
    return

  # Get arity and max column width
  arity = len(relation[0])

  # Check if the tuple is empty
  if arity == 0 and not ctx.is_probabilistic():
    relation[0] = ("()",)
  elif arity == 1 and ctx.is_probabilistic():
    relation[0] = (relation[0][0], "()")

  # Compute column width
  max_column_width = [MAX_COL_WIDTH for _ in range(arity)]

  # Then get the headers
  headers = ctx.relation_field_names(relation_name)
  if headers and ctx.is_probabilistic():
    headers = ["prob"] + headers
    max_column_width = [None] + max_column_width

  # Check if we need multi-line table
  need_multi_line_table = False
  for tup in relation:
    for val in tup:
      if type(val) == str and len(val) > MAX_COL_WIDTH:
        need_multi_line_table = True

  table_fmt = MULTILINE_TABLE_FMT if need_multi_line_table else TABLE_FMT
  if headers: to_print = tabulate(relation, headers=headers, tablefmt=table_fmt, maxcolwidths=max_column_width)
  else: to_print = tabulate(relation, tablefmt=table_fmt, maxcolwidths=max_column_width)
  print(to_print)


def interpret(ctx, args):
  try:
    ctx.import_file(args.file)
  except Exception as e:
    print(e, file=sys.stderr)

  # Run the context
  ctx.run()

  # Print the results
  to_output_relations = [r for r in ctx.relations() if ctx.has_relation(r)]
  for relation in sorted(to_output_relations):
    print(f"{relation}:")
    print_relation(ctx, relation)


def repl(ctx, args):
  # If module is specified in args, load the module
  if args.module is not None:
    ctx.import_file(args.module)

  # Create a prompt session
  prompt = prompt_toolkit.PromptSession()
  while True:
    try:
      user_input = prompt.prompt('scl> ')
      queries = ctx.add_item(user_input)
      if len(queries) > 0:
        ctx.run()
        for query in queries:
          if len(queries) > 1:
            print(f"{query}:")
          print_relation(ctx, query)
    except EOFError:
      exit()
    except KeyboardInterrupt:
      exit()
    except Exception as err:
      print(err, file=sys.stderr)


def setup_context(ctx: scallopy.ScallopContext, args):
  # Iteration limit
  ctx.set_iter_limit(args.iter_limit)

  # Debug
  if args.debug_front:
    ctx.set_debug_front()
  if args.debug_back:
    ctx.set_debug_back()
  if args.debug_ram:
    ctx.set_debug_ram()


def main():
  plugin_registry = scallopy.PluginRegistry()

  # Parse command line arguments
  args, unknown_args = cmd_args(plugin_registry)

  # Configure environments
  plugin_registry.configure(args, unknown_args)
  if args.dump_loaded_plugins:
    plugin_registry.dump_loaded_plugins()

  # Create a scallopy context
  ctx = scallopy.ScallopContext(
    provenance=args.provenance,
    k=args.top_k,
    wmc_with_disjunctions=args.wmc_with_disjunctions)
  setup_context(ctx, args)

  # Load the scallopy extension library
  plugin_registry.load_into_ctx(ctx)

  # Check if the user has provided a file
  if args.file is not None:
    # If so, interpret the file directly
    interpret(ctx, args)
  else:
    # Otherwise, enter REPL
    repl(ctx, args)
