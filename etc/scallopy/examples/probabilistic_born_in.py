import scallopy

ctx = scallopy.ScallopContext(provenance="minmaxprob")

ctx.add_relation("speaks", (int, str))
ctx.add_facts("speaks", [
  (None, (0, "english")),
  (None, (1, "chinese")),
  (None, (2, "english")),
])

ctx.add_relation("lives_in", (int, str))
ctx.add_facts("lives_in", [
  (None, (0, "china")),
  (None, (1, "us")),
  (None, (2, "us")),
])

# If one lives in China, one might be borned in China (0.8)
ctx.add_rule("born_in(a, \"china\") = lives_in(a, \"china\")", tag = 0.8)

# If one speaks Chinese, one might be borned in China (0.6)
ctx.add_rule("born_in(a, \"china\") = speaks(a, \"chinese\")", tag = 0.6)

ctx.run()

print("born_in:", list(ctx.relation("born_in")))
