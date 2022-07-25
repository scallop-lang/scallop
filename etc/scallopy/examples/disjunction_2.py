import scallopy

ctx = scallopy.ScallopContext(provenance="proofs")
ctx.add_relation("constant", (str, int))
ctx.add_relation("plus_expr", (str, str, str))
ctx.add_relation("root", str)
ctx.add_rule("eval(e, y) = constant(e, y)")
ctx.add_rule("eval(e, x + y) = plus_expr(e, l, r), eval(l, x), eval(r, y)")
ctx.add_rule("result(e, y) = root(e), eval(e, y)")

# 1 + 2 + 3
# - - - - -
# A B C D E
ctx.add_facts("constant", [("A", 1), ("C", 2), ("E", 3)])

# Possible expressions:

# Expr 1
#    +
#   /  \
#  1     +
#       / \
#      2   3
expr_1 = [("B", "A", "D"), ("D", "C", "E")]

# Expr 2 (Correct)
#        +
#      /  \
#    +     3
#   / \
#  1   2
expr_2 = [("D", "B", "E"), ("B", "A", "C")]

# Expression facts -- note that if the last `[[0, 2]]` is not added,
# the program will fall into an infinite loop
ctx.add_facts("plus_expr", expr_1 + expr_2, disjunctions=[[0, 2]])

# Root could be "B" or "D"
ctx.add_facts("root", [("B",), ("D",)])

# Run
ctx.run()

print(list(ctx.relation("result")))
