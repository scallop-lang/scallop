from .utils import _mapping_tuple, _map_entity_tuple_to_str_tuple

class OutputMapping:
  def __init__(self, output_mapping):
    if type(output_mapping) == list:
      self.is_none = False
      self.singleton = False
      self.mapping = {0: [_mapping_tuple(t) for t in output_mapping]}
      self.shape = (len(self.tuples),)
    elif type(output_mapping) == tuple:
      self.is_none = False
      self.singleton = True
      self.mapping = {0: [_mapping_tuple(output_mapping)]}
      self.shape = (1,)
    elif type(output_mapping) == range:
      self.is_none = False
      self.singleton = False
      self.mapping = {0: [_mapping_tuple(t) for t in list(output_mapping)]}
      self.shape = (len(self.tuples),)
    elif type(output_mapping) == dict:
      num_dim = len(output_mapping)
      for i in range(num_dim):
        assert i in output_mapping, f"Non-existed dimension {i} in output mapping"
      self.mapping = {key: [_mapping_tuple(t) for t in tuples] for (key, tuples) in output_mapping}
      self.is_none = False
      self.singleton = False
      self.dimensional_mapping
    elif output_mapping is None:
      self.is_none = True
    else:
      raise Exception(f"Unknown output mapping type `{type(output_mapping)}`")

  def dim(self):
    assert not self.is_none, "Cannot obtain dimension from a `None` output mapping"
    return len(self.shape)
