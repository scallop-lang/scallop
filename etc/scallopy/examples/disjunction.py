import scallopy

ctx = scallopy.ScallopContext(provenance="proofs")
ctx.add_relation("digit", (int, int))
ctx.add_facts("digit", [(0, i) for i in range(10)], disjunctions=[list(range(10))])
ctx.add_facts("digit", [(1, i) for i in range(10)], disjunctions=[list(range(10))])
ctx.add_rule("sum2(x, y, a + b) = digit(x, a), digit(y, b)")
ctx.run()
print(list(ctx.relation("sum2")))
