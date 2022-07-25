import scallopy

class CustomProbabilityProvenance(scallopy.ScallopProvenance):
  def base(self, p): return p if p is not None else 1.0

  def is_valid(self, t): return True

  def zero(self): return 0.0

  def one(self): return 1.0

  def add(self, t1, t2): return t1 + t2

  def mult(self, t1, t2): return t1 * t2

  def aggregate_unique(self, elems):
    max_prob = 0.0
    max_elem = None
    for (prob, tup) in elems:
      if prob > max_prob:
        max_prob = prob
        max_elem = tup
    if max_elem: return [(max_prob, max_elem)]
    else: return []


prov_ctx = CustomProbabilityProvenance()
ctx = scallopy.ScallopContext(provenance="custom", custom_provenance=prov_ctx)
ctx.add_relation("digit_1", int)
ctx.add_relation("digit_2", int)
ctx.add_rule("sum_2(a + b) = digit_1(a), digit_2(b)")
ctx.add_rule("result(x) = x = unique(x: sum_2(x))")

# Add facts
ctx.add_facts("digit_1", [(0.1, (1,)), (0.2, (2,)), (0.3, (3,))])
ctx.add_facts("digit_2", [(0.1, (4,)), (0.2, (5,)), (0.3, (6,))])

# Run
ctx.run()

# Inspect result
print(list(ctx.relation("result")))
