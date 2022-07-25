from typing import Optional

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
  ):
    self.path = path
    self.deliminator = deliminator
    self.has_header = has_header
    self.has_probability = has_probability
