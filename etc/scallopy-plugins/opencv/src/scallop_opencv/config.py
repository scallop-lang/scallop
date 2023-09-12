from argparse import ArgumentParser


_DEFAULT_SAVE_IMAGE_PATH = ".tmp/scallop-save-image"
SAVE_IMAGE_PATH = _DEFAULT_SAVE_IMAGE_PATH


def setup_arg_parser(parser: ArgumentParser):
  parser.add_argument("--save-image-path", type=str, default=_DEFAULT_SAVE_IMAGE_PATH)


def configure(args):
  global SAVE_IMAGE_PATH
  SAVE_IMAGE_PATH = args.save_image_path
