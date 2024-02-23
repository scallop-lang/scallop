from typing import Optional
import os
import subprocess
import re
import argparse

PY_PROJECT_SRC = """\
[project]
name = "scallop-{dashed_name}"
version = "{version}"
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]

[project.entry-points."scallop.plugin.setup_arg_parser"]
{dashed_name} = "scallop_{underscore_name}:setup_arg_parser"

[project.entry-points."scallop.plugin.configure"]
{dashed_name} = "scallop_{underscore_name}:configure"

[project.entry-points."scallop.plugin.load_into_context"]
{dashed_name} = "scallop_{underscore_name}:load_into_context"
"""


MAKEFILE_SRC = """\
develop:
\tpip install --editable .

install: build
\tfind dist -name "*.whl" -print | xargs pip install --force-reinstall

build:
\tpython -m build

clean:
\trm -rf dist
"""


GITIGNORE_SRC = """\
*.egg-info
dist
"""


INIT_SRC = """\
import argparse
import scallopy

def setup_arg_parser(parser: argparse.ArgumentParser):
  pass

def configure(args):
  pass

def load_into_context(ctx: scallopy.Context):
  pass
"""


DONE_PRINT = """\
Initialized empty Scallop plugin in {path}"""


ALLOWED_DIR_OBJS = set([
  ".DS_Store",
  "Makefile",
  "makefile",
  "readme.md",
  "Readme.md",
  "readme",
  "Readme",
  ".vscode",
  ".git",
  ".github",
])


def do_prompt_and_get_answer(
    prompt: str,
    require_non_empty: bool = True,
    default_value: Optional[str] = None,
    pattern: Optional[re.Pattern[str]] = None,
    empty_string_print: str = "[Error] The answer cannot be empty; please try again",
    pattern_fail_print: str = "[Error] Does not match the pattern {pattern}",
    failed_print: str = "[Error] Failed to get a proper answer; please try `create-scallopy-plugin` again",
    trim_whitespace: bool = True,
    num_try: Optional[int] = 3
):
  try_count = 0
  while num_try is None or try_count < num_try:
    try_count += 1
    ans = input(prompt)
    if trim_whitespace:
      ans = ans.strip()
    if ans == "":
      if require_non_empty:
        print(empty_string_print)
      else:
        return default_value
    else:
      if re.search(pattern, ans) is not None:
        return ans
      else:
        print(pattern_fail_print.format(pattern=str(pattern)))
  print(failed_print)
  exit(1)


def is_git_repo():
  """Check whether this is a git repo"""
  with open(os.devnull, "w") as d:
    return not bool(subprocess.call('git rev-parse', shell=True, stdout=d, stderr=d))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("create-plugin")
  parser.add_argument("path", nargs="?", default=None)
  args = parser.parse_args()

  # Check whether path is provided in the arguments
  if args.path is not None:
    subprocess.run(["mkdir", "-p", args.path])
    cwd = args.path
  else:
    cwd = "."

  # Check if the current folder does not contain `pyproject.toml`, `makefile`, and `src`
  curr_dir_objs = set(os.listdir(cwd))
  disallowed_dir_objs = curr_dir_objs.difference(ALLOWED_DIR_OBJS)
  if len(disallowed_dir_objs) > 0:
    print(f"Cannot create a scallopy plugin in non-empty directory. Found object `{disallowed_dir_objs.pop()}`")
    exit(2)

  # Collect essential information
  print("This script will guide you creating a new scallopy plugin.")
  print("Please help answer the following questions:")
  name = do_prompt_and_get_answer("Plugin Name: scallop-", pattern=r"^[a-zA-Z][a-zA-Z\-0-9]*$")
  version = do_prompt_and_get_answer("Version (1.0.0): ", pattern=r"^\d+\.\d+\.\d+$", require_non_empty=False, default_value="1.0.0")

  # Run git init
  if not is_git_repo():
    with open(os.devnull, "w") as d:
      subprocess.call("git init", shell=True, stdout=d, stderr=d, cwd=cwd)

  # Write .gitignore file
  with open(os.path.join(cwd, ".gitignore"), "w") as gitignore_file:
    gitignore_file.write(GITIGNORE_SRC)

  # Create pyproject.toml
  underscore_name = name.replace("-", "_")
  py_project_src = PY_PROJECT_SRC.format(dashed_name=name, underscore_name=underscore_name, version=version)
  with open(os.path.join(cwd, "pyproject.toml"), "w") as pyproject_file:
    pyproject_file.write(py_project_src)

  # Create makefile
  with open(os.path.join(cwd, "makefile"), "w") as makefile_file:
    makefile_file.write(MAKEFILE_SRC)

  # Create folders
  subprocess.run(["mkdir", "-p", f"src/scallop_{name}"], cwd=cwd)
  with open(os.path.join(cwd, "src", f"scallop_{name}", "__init__.py"), "w") as init_file:
    init_file.write(INIT_SRC)

  # Final prints
  print(DONE_PRINT.format(path=os.path.abspath(os.path.join(cwd))))
