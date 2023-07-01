from typing import Optional, List, Union
from copy import deepcopy


class CSVFileOptions:
  """
  CSV File Options

  :param path, the path to the CSV file
  :param deliminator, the deliminator of the CSV file. Default to `","`
  :param has_header, whether the CSV has header. Default to `False`
  :param has_probability, whether the CSV has probability. With probability, the first column should be CSV. Default to `False`
  """
  def __init__(
    self,
    path: str,
    deliminator: Optional[str] = None,
    has_header: bool = False,
    has_probability: bool = False,
    keys: Optional[Union[List[str], str]] = None,
    fields: Optional[List[str]] = None,
  ):
    # Basic properties
    self.path = path
    self.deliminator = deliminator
    self.has_probability = has_probability

    # Sanitize the keys and fields
    self.keys = _sanitize_keys(keys)
    self.fields = fields

    # If there is key or field, then there has to be header
    self.has_header = has_header or (self.keys is not None) or (self.fields is not None)

  def with_deliminator(self, deliminator: str):
    copied = deepcopy(self)
    copied.deliminator = deliminator
    return copied

  def with_fields(self, fields: Optional[List[str]]):
    copied = deepcopy(self)
    copied.fields = fields
    return copied

  def with_keys(self, keys: Optional[Union[str, List[str]]]):
    copied = deepcopy(self)
    copied.keys = keys
    return copied


def _sanitize_keys(keys: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
  if keys is not None:
    if type(keys) is str:
      return [keys]
    else:
      assert type(keys) is list, "`keys` should be a string or a list of strings"
      for elem in keys:
        assert type(elem) is str, "an element in `keys` should be a string"
      return keys
  else:
    return None
