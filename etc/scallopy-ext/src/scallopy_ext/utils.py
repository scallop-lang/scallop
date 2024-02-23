from typing import Callable, Optional, Dict, List, Union
from importlib import metadata
import argparse


def _process_args(args):
  if isinstance(args, argparse.Namespace):
    return {str(key): args.__dict__[key] for key in args.__dict__.keys()}
  elif type(args) == dict:
    return args
  else:
    return {}


def _find_entry_points(
    entry_points: Dict[str, List[metadata.EntryPoint]],
    group: str,
    filter: Optional[Callable[[str], bool]] = None,
):
  if group in entry_points:
    eps = list(set(entry_points[group]))
    if filter is not None:
      eps = [ep for ep in eps if filter(ep.name)]
    return eps
  else:
    return []


def _dedup_extend(array: List, new_array: List):
  for elem in new_array:
    if elem not in array:
      array.append(elem)


class CustomEntryPoint:
  """
  A custom entry point as a replacement of `importlib.metadata` entry point
  """
  def __init__(self, name: str, fn_name: str, fn: Callable):
    self.name = name
    self.group = f"scallop.plugin.{fn_name}"
    self.fn = fn

  def load(self):
    return self.fn


EntryPoint = Union[CustomEntryPoint, metadata.EntryPoint]
