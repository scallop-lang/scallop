# Map any element into a tuple
# - If the element is a scalar, turn that into an arity-1 tuple
# - Otherwise return the tuple directly
def _mapping_tuple(t):
  return t if type(t) == tuple else (t,)
