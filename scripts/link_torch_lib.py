import os
import subprocess
import re
import argparse

# Get an argument
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

# First get the python directory
output = subprocess.run(["which", "python"], stdout=subprocess.PIPE).stdout.decode()
python_dir = output.split("\n")[0]
if args.verbose:
  print(f"Python Directory: {python_dir}")

# Check python version
python_version = subprocess.run([python_dir, "--version"], stdout=subprocess.PIPE).stdout.decode()
capture = re.search('Python (\\d+).(\\d+).(\\d+)', python_version)
version = f"{capture[1]}.{capture[2]}"
if args.verbose:
  print(f"Python Version: {version}")

# Get the lib
lib_dir = os.path.abspath(os.path.join(python_dir, "..", "..", "lib", f"python{version}", "site-packages", "torch"))
if not os.path.exists(lib_dir):
  print(f"[Error] Torch lib `{lib_dir}` does not exist")
if args.verbose:
  print(f"Torch Lib: {lib_dir}")

# Link the library to the current directory
tmp_dir = os.path.join(os.path.dirname(__file__), "..", ".tmp")
if not os.path.exists(tmp_dir):
  os.mkdir(tmp_dir)
ln_dir = os.path.abspath(os.path.join(tmp_dir, "torch"))
if args.verbose:
  print(f"To be linked location: {ln_dir}")
subprocess.run(["ln", "-sf", lib_dir, ln_dir])
