from scallopy import ScallopContext

# Create new context
ctx = ScallopContext()

# Construct the program
ctx.add_relation("edge", (int, int))
ctx.add_facts("edge", [(0, 1), (1, 2), (2, 3)])
ctx.add_rule("path(a, c) = edge(a, c)")
ctx.add_rule("path(a, c) = edge(a, b), path(b, c)")

# Run the program
ctx.run()

# Inspect the result
paths = ctx.relation("path")
for (s, t) in paths:
  print(f"elem: ({s}, {t})")
