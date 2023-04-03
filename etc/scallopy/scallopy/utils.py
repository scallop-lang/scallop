# Map any element into a tuple
# - If the element is a scalar, turn that into an arity-1 tuple
# - Otherwise return the tuple directly
def _mapping_tuple(t):
  return t if type(t) == tuple else (t,)


class Counter:
  def __init__(self):
    self.count = 0

  def get_and_increment(self) -> int:
    result = self.count
    self.count += 1
    return result
