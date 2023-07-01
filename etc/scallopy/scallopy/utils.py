# Map any element into a tuple
# - If the element is a scalar, turn that into an arity-1 tuple
# - Otherwise return the tuple directly
def _mapping_tuple(t):
  return t if type(t) == tuple else (t,)


def _map_entity_to_str(entity):
  if type(entity) == bool:
    return "true" if entity else "false"
  elif type(entity) == int or type(entity) == float:
    return str(entity)
  elif type(entity) == str:
    return entity
  else:
    raise Exception(f"Unknown entity type {type(entity)}")


def _map_entity_tuple_to_str_tuple(entity):
  if type(entity) is tuple or type(entity) is list:
    return tuple([_map_entity_to_str(element) for element in entity])
  else:
    entity_element = _map_entity_to_str(entity)
    return (entity_element,)


class Counter:
  def __init__(self):
    self.count = 0

  def get_and_increment(self) -> int:
    result = self.count
    self.count += 1
    return result
