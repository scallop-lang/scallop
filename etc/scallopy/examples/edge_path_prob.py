from scallopy import ScallopContext

# Create new context
ctx = ScallopContext(provenance = "minmaxprob")

# Construct the program
ctx.add_relation("edge", (int, int))
ctx.add_facts("edge", [(0.1, (0, 1)), (0.2, (1, 2)), (0.3, (2, 3))])
ctx.add_rule("path(a, c) = edge(a, c)")
ctx.add_rule("path(a, c) = edge(a, b), path(b, c)")

# Run the program
ctx.run()

# Inspect the result
paths = ctx.relation("path")
for (p, (s, t)) in paths:
  print(f"prob: {p}, elem: ({s}, {t})")
