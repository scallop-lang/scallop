from typing import *

import scallopy
from scallopy import foreign_predicate, Facts


@foreign_predicate
def string_semantic_eq(s1: str, s2: str) -> Facts[float, Tuple]:
  if s1 == "mom" and s2 == "mother":
    yield (0.99, ())
  elif s1 == "mother" and s2 == "mother":
    yield (1.0, ())


ctx = scallopy.Context(provenance="minmaxprob")
ctx.register_foreign_predicate(string_semantic_eq)
ctx.add_relation("kinship", (str, str, str))
ctx.add_facts("kinship", [
  (1.0, ("alice", "mom", "bob")),
  (1.0, ("alice", "mother", "cassey")),
])
ctx.add_rule("parent(a, b) = kinship(a, r, b) and string_semantic_eq(r, \"mother\")")
ctx.add_rule("sibling(a, b) = parent(c, a) and parent(c, b) and a != b")
ctx.run()
print("kinship", list(ctx.relation("kinship")))
print("sibling", list(ctx.relation("sibling")))
print("parent", list(ctx.relation("parent")))
