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
