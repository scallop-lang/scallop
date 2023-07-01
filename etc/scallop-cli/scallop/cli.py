# Argument parser
import argparse

# Scallop imports
import scallopy

# Project imports
from . import config
from . import stdlib


def argument_parser():
  parser = argparse.ArgumentParser("scallop", description="Scallop language command line interface")
  parser.add_argument("file", nargs="?", default=None, help="The file to execute")
  parser.add_argument("-p", "--provenance", type=str, default="unit", help="The provenance to pick")
  parser.add_argument("-m", "--module", type=str, default=None, help="Load module in interactive mode")
  parser.add_argument("--iter-limit", type=int, default=10, help="Iteration limit")
  parser.add_argument("--num-allowed-openai-request", type=int, default=100, help="Limit on the number of openai calls")
  parser.add_argument("--openai-gpt-model", type=str, default="text-davinci-003", help="The GPT model we use")
  parser.add_argument("--openai-gpt-temperature", type=float, default=0, help="The temperature for the GPT model")
  return parser


def cmd_args():
  parser = argument_parser()
  return parser.parse_args()


def main():
  # Parse command line arguments
  args = cmd_args()

  # Configure environments
  config.configure(args)

  # Create a scallopy context
  ctx = scallopy.ScallopContext(provenance=args.provenance)
  ctx.set_iter_limit(args.iter_limit)

  # Load the stdlib
  stdlib.load_stdlib(ctx)

  # Check if the user has provided a file
  if args.file is not None:
    # When file is available, Import the target file into the context
    ctx.import_file(args.file)

    # Run the context
    ctx.run()

    # Print the results
    for relation in ctx.relations():
      if ctx.has_relation(relation):
        print(f"{relation}\t:\t{list(ctx.relation(relation))}")

  else:
    raise NotImplementedError()
