import scallopy
import torch

# Build up scallop context
ctx = scallopy.ScallopContext(provenance="difftopkproofs")
ctx.add_relation("constant", (str, int))
ctx.add_relation("plus_expr", (str, str, str))
ctx.add_relation("root", str)
ctx.add_rule("eval(e, y) = constant(e, y)")
ctx.add_rule("eval(e, x + y) = plus_expr(e, l, r), eval(l, x), eval(r, y)")
ctx.add_rule("result(e, y) = root(e), eval(e, y)")

# Generate forward function
eval_expr = ctx.forward_function("result")

# Data for execution
constant = [[(torch.tensor(0.5), ("A", 1)), (torch.tensor(0.5), ("C", 2)), (torch.tensor(0.5), ("E", 3))]]
plus_expr = [[(torch.tensor(0.8), ("B", "A", "D")), (torch.tensor(0.8), ("D", "C", "E")), (torch.tensor(0.2), ("D", "B", "E")), (torch.tensor(0.2), ("B", "A", "C"))]]
root = [[(torch.tensor(1.0), ("B",)), (torch.tensor(1.0), ("D",))]]
disjunctions = {"plus_expr": [[[0, 2]]]}

# Evaluate!
mapping, tensor = eval_expr(constant=constant, plus_expr=plus_expr, root=root, disjunctions=disjunctions)
print(mapping)
print(tensor)
