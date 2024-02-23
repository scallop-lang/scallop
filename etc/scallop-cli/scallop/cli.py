import argparse

def main():
  parser = argparse.ArgumentParser("scallop", description="Scallop language command line interface")
  parser.add_argument("first_arg", nargs="?", default=None)
  args, _ = parser.parse_known_args()
  if args.first_arg == "create-plugin":
    from .create_plugin import main as create_plugin_main
    create_plugin_main()
  else:
    from .run_scallop import main as run_scallop_main
    run_scallop_main()
