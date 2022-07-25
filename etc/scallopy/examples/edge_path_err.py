from scallopy import ScallopContext

# Create new context
ctx = ScallopContext(provenance = "unit")

# Construct the program
ctx.add_relation("edge", (int, int))

# Add tuples that are having wrong type
ctx.add_facts("edge", [(0, 1), (1, 2), (2, "ERROR")])
