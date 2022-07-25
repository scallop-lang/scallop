import scallopy

class CustomProbabilityProvenance(scallopy.ScallopProvenance):
  def base(self, p):
    if p is not None:
      return p
    else:
      return 1.0

  def disjunction_base(self, infos):
    return [1.0 if i is None else i for i in infos]

  def is_valid(self, t):
    return True

  def zero(self):
    return 0.0

  def one(self):
    return 1.0

  def add(self, t1, t2):
    return t1 + t2

  def mult(self, t1, t2):
    return t1 * t2

prov_ctx = CustomProbabilityProvenance()
ctx = scallopy.ScallopContext(provenance="custom", custom_provenance=prov_ctx)
ctx.add_relation("edge", (int, int))
ctx.add_facts("edge", [
  (0.5, (0, 1)),
  (0.5, (1, 2)),
  (0.5, (2, 3)),
])
ctx.add_rule("path(a, b) = edge(a, b)")
ctx.add_rule("path(a, b) = path(a, c), edge(c, b)")
ctx.run()
print(list(ctx.relation("path")))
