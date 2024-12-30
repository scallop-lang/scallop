import os
from argparse import ArgumentParser
from typing import Dict, List, Any
import yaml
import csv
import subprocess

import scallopy

from .utils.constants import CUSTOM_CODEQL_QUERY_DIR

SUPPORTED_LANGUAGES = {
  "java": {
    "custom_script_path": "qlpacks/codeql/java-queries/0.8.3/scallop_codeql",
  }
}

class ScallopCodeQLPlugin(scallopy.Plugin):
  def __init__(self):
    super().__init__()

    # Configurations
    self._configured = False
    self._codeql_path = None
    self._codeql_exec = None

    # Project specific
    self._codeql_db = None
    self._codeql_db_yaml_dir = None
    self._codeql_db_yaml = None
    self._project_language = None

  def setup_argparse(self, parser: ArgumentParser):
    parser.add_argument("--codeql-db", type=str, help="The CodeQL database of the target repository that we are going to analyze")
    parser.add_argument("--codeql-path", type=str, help="The directory to the `codeql` folder. The `codeql` executable should be within the folder")

  def configure(self, args: Dict = ..., unknown_args: List = ...):
    # First check if the user has specified a codeql DB or not
    self._codeql_db = args["codeql_db"]
    if self._codeql_db is None:
      # Ignore
      return
    else:
      # Check that the database is well formed
      self._codeql_db_yaml_dir = f"{self._codeql_db}/codeql-database.yml"
      if not os.path.exists(self._codeql_db_yaml_dir):
        raise Exception("Incomplete CodeQL database; expected codeql-database.yml within the db directory")

      # Load the yaml file
      self._codeql_db_yaml = yaml.safe_load(open(self._codeql_db_yaml_dir))
      if not self._codeql_db_yaml["finalised"]:
        raise Exception("Incomplete CodeQL database; it is not finalised")

      # Check the language of the project
      self._project_language = self._codeql_db_yaml["primaryLanguage"]
      if self._project_language not in SUPPORTED_LANGUAGES:
        raise Exception(f"Unsupported project language `{self._project_language}`")

      # Get the source code directory of the project
      self._project_source_code_dir = self._codeql_db_yaml["sourceLocationPrefix"]

      # Create a temporary directory for Scallop-CodeQL
      self._cache_dir = os.path.join(self._codeql_db, "scallop_codeql", self._project_language)
      os.makedirs(self._cache_dir, exist_ok=True)

    # If there is a CodeQL DB from the arguments, we need to
    self._codeql_path = None
    env_codeql_path = os.getenv("CODEQL_PATH")
    if env_codeql_path is not None:
      self._codeql_path = env_codeql_path
    elif "codeql_path" in args:
      self._codeql_path = args["codeql_path"]
    else:
      raise Exception("Missing CodeQL path; either set the environment variable CODEQL_PATH or pass as a command line argument `--codeql-path`")

    # Check if the codeql executable is there
    self._codeql_exec = f"{self._codeql_path}/codeql"
    if not os.path.exists(self._codeql_exec):
      raise Exception(f"[scallop_codeql] `codeql` executable not found under CODEQL_PATH: {self._codeql_path}")

    # At the end, the codeql plugin has been successfully initialized
    self._configured = True

  def load_into_ctx(self, ctx: scallopy.ScallopContext):
    # Register each predicate one-by-one
    from .attributes.codeql_database import get_codeql_database
    ctx.register_foreign_attribute(get_codeql_database(self))

    from .attributes.codeql_source import get_codeql_source
    # ctx.register_foreign_attribute(get_codeql_source(self))

  def project_language(self) -> str:
    return self._project_language

  def is_java(self) -> bool:
    return self._project_language == "java"

  def run_codeql_query(self, query: str, output: str, overwrite: bool = False, debug: bool = False) -> str:
    """
    Run the given CodeQL query

    :param query, a relative directory from `CUSTOM_CODEQL_QUERY_DIR`
    :param output, the output file name (without bprs or csv)
    """

    # Directories
    query_file_dir = f"{CUSTOM_CODEQL_QUERY_DIR}/{query}"
    target_file_dir = f"{self._codeql_path}/{SUPPORTED_LANGUAGES[self._project_language]['custom_script_path']}/{query}"
    result_bqrs_file_dir = f"{self._cache_dir}/{output}.bqrs"
    result_csv_file_dir = f"{self._cache_dir}/{output}.csv"

    # Found in cache
    if os.path.exists(result_csv_file_dir) and not overwrite:
      if debug: print(f"[scallop_codeql] result CSV `{result_csv_file_dir}` found; returning directly")
      return result_csv_file_dir

    # 1. Copy the query to CodeQL directory
    target_file_parent_dir = os.path.dirname(target_file_dir)
    if not os.path.exists(target_file_parent_dir):
      if debug: print(f"[scallop_codeql] target file directory not exists; creating directory `{target_file_parent_dir}`")
      os.makedirs(target_file_parent_dir, exist_ok=True)
    cp_cmd = ["cp", query_file_dir, target_file_dir]
    if debug: print(f"[scallop_codeql] copying query file to CodeQL directory `{target_file_dir}`")
    subprocess.run(cp_cmd, stderr=subprocess.PIPE)
    assert os.path.exists(target_file_dir), f"[scallop_codeql] Internal Error; CodeQL query not copied successfully"

    # 2. Run the query and output to bprs
    codeql_run_query_cmd = [self._codeql_exec, "query", "run", f"--database={self._codeql_db}", f"--output={result_bqrs_file_dir}", "--", target_file_dir]
    if debug: print(f"[scallop_codeql] running CodeQL query...")
    run_query_output = subprocess.run(codeql_run_query_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert os.path.exists(result_bqrs_file_dir), f"[scallop_codeql] CodeQL Error: CodeQL failed to produce analysis result bqrs\n\nCodeQL STDOUT:\n{run_query_output.stderr}"
    if debug: print(f"[scallop_codeql] success; obtained bqrs file `{result_bqrs_file_dir}`")

    # 3. Convert the output to csv
    codeql_decode_bqrs_cmd = [self._codeql_exec, "bqrs", "decode", result_bqrs_file_dir, "--format=csv", f"--output={result_csv_file_dir}"]
    if debug: print(f"[scallop_codeql] running CodeQL decode...")
    decode_output = subprocess.run(codeql_decode_bqrs_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert os.path.exists(result_csv_file_dir), f"[scallop_codeql] CodeQL Error: CodeQL failed to decode bqrs into csv\n\nCodeQL STDOUT:\n{decode_output.stderr}"
    if debug: print(f"[scallop_codeql] success; obtained csv file `{result_csv_file_dir}`")

    # 4. Return the csv
    return result_csv_file_dir

  def try_load_index_csv(self, index_csv_name: str) -> str:
    csv_file_dir = f"{self._cache_dir}/{index_csv_name}.csv"
    if os.path.exists(csv_file_dir):
      return csv_file_dir
    else:
      return None

  def save_index_csv(self, index_csv_name: str, fields: List[str], data: List[List[Any]]) -> str:
    csv_file_dir = f"{self._cache_dir}/{index_csv_name}.csv"
    csv_writer = csv.writer(open(csv_file_dir, "w"))
    csv_writer.writerow(fields)
    csv_writer.writerows(data)
    return csv_file_dir
