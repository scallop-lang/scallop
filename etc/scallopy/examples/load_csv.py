import os
from scallopy import *

ctx = ScallopContext()

# Add relation with load_csv
this_path = os.path.dirname(os.path.realpath(__file__))
edge_csv_path = os.path.join(this_path, "../../../examples/input_csv/edge.csv")
ctx.add_relation("edge", (int, int), load_csv=edge_csv_path)

# Add a rule
ctx.add_rule("path(a, b) = edge(a, b) or (path(a, c) and edge(c, b))")

# Run
ctx.run()

# Inspect the result
print("edge:", list(ctx.relation("edge")))
print("path:", list(ctx.relation("path")))
