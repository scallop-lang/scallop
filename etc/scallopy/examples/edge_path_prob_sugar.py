from scallopy import ScallopContext
from scallopy.sugar import Relation

# Create new context
ctx = ScallopContext(provenance="minmaxprob")

# Construct the program

edge = Relation(ctx, (int, int))
path = Relation(ctx, (int, int))
edge |= [(0.1, (0, 1)), (0.2, (1, 2)), (0.3, (2, 3))]
path["a", "c"] |= edge["a", "c"]
path["a", "c"] |= edge["a", "b"] & path["b", "c"]

# Run the program
ctx.run()

# Inspect the result
for (p, (s, t)) in path:
  print(f"prob: {p}, elem: ({s}, {t})")
