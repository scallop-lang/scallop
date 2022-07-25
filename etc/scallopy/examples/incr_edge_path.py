import scallopy

ctx = scallopy.ScallopContext()

print("Start 0")
ctx.add_relation("edge", (int, int))
ctx.add_facts("edge", list([(i, i + 1) for i in range(1000)]))
print("Run 0")
ctx.run()
print("End phase 0")

# Compute undir path using the first method
print("Start 1")
ctx1 = ctx.clone()
ctx1.add_rule("undir_edge(a, b) = edge(a, b)")
ctx1.add_rule("undir_edge(a, b) = edge(b, a)")
ctx1.add_rule("undir_path(a, b) = undir_edge(a, b) or (undir_path(a, c) and undir_edge(c, b))")
print("Run 1")
ctx1.run()
print("End phase 1")

# Compute undir path using the second method
print("Start 2")
ctx2 = ctx.clone()
ctx2.add_rule("path(a, b) = edge(a, b) or (edge(a, c) and path(c, b))")
ctx2.add_rule("undir_path(a, b) = path(a, b) or path(b, a)")
print("Run 2")
ctx2.run()
print("End phase 2")

# Count how many paths and undirected paths in ctx2
print("Start 3")
ctx2.add_rule("how_many_path(n) = n = count(a, b: path(a, b))")
ctx2.add_rule("how_many_undir_path(n) = n = count(a, b: undir_path(a, b))")
print("Run 3")
ctx2.run()
print("End phase 3")
print("how many paths:", list(ctx2.relation("how_many_path")))
print("how many undirected paths:", list(ctx2.relation("how_many_undir_path")))
