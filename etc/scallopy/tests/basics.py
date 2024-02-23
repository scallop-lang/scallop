import unittest

import scallopy

class BasicTests(unittest.TestCase):
  def test_edge_path(self):
    ctx = scallopy.ScallopContext()
    ctx.add_relation("edge", (int, int))
    ctx.add_facts("edge", [(1, 2), (2, 3), (3, 4)])
    ctx.add_rule("path(a, c) = edge(a, c) or (path(a, b) and edge(b, c))")
    ctx.run()
    self.assertEqual(list(ctx.relation("path")), [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])

  def test_fibonacci(self):
    ctx = scallopy.ScallopContext()
    ctx.add_relation("fib", (int, int))
    ctx.add_facts("fib", [(0, 1), (1, 1)])
    ctx.add_rule("fib(x, y + z) = fib(x - 1, y), fib(x - 2, z), x <= 5")
    ctx.run()
    self.assertEqual(list(ctx.relation("fib")), [(0, 1), (1, 1), (2, 2), (3, 3), (4, 5), (5, 8)])

  def test_fibonacci_demand(self):
    ctx = scallopy.ScallopContext()
    ctx.add_relation("fib", (int, int))
    ctx.add_facts("fib", [(0, 1), (1, 1)])
    ctx.add_rule("fib(x, y + z) = fib(x - 1, y), fib(x - 2, z), x > 1", demand="bf")
    ctx.add_rule("result(y) = fib(5, y)")
    ctx.run()
    self.assertEqual(list(ctx.relation("result")), [(8,)])

  def test_digit_sum(self):
    ctx = scallopy.ScallopContext()
    ctx.add_relation("digit1", int)
    ctx.add_relation("digit2", int)
    ctx.add_facts("digit1", [(i,) for i in range(10)])
    ctx.add_facts("digit2", [(i,) for i in range(10)])
    ctx.add_rule("sum_2(a + b) = digit1(a), digit2(b)")
    ctx.run()
    self.assertEqual(list(ctx.relation("sum_2")), [(i,) for i in range(19)])

  def test_softmax(self):
    ctx = scallopy.ScallopContext(provenance="minmaxprob")
    ctx.add_relation("digit", int)
    ctx.add_facts("digit", [(0.5, (i,)) for i in range(10)])
    ctx.add_rule("softmax_digit(n) = n := softmax(d: digit(d))")
    ctx.run()
    for (i, (prob, (p_i,))) in enumerate(list(ctx.relation("softmax_digit"))):
      self.assertAlmostEqual(prob, 0.1)
      self.assertEqual(i, p_i)

  def test_rule_weight(self):
    ctx = scallopy.ScallopContext(provenance="minmaxprob")
    ctx.add_relation("speaks", (int, str))
    ctx.add_relation("lives_in", (int, str))
    ctx.add_facts("speaks", [(None, (0, "english")), (None, (1, "chinese")), (None, (2, "english"))])
    ctx.add_facts("lives_in", [(None, (0, "china")), (None, (1, "us")), (None, (2, "us"))])
    ctx.add_rule("born_in(a, \"china\") = lives_in(a, \"china\")", tag = 0.8)
    ctx.add_rule("born_in(a, \"china\") = speaks(a, \"chinese\")", tag = 0.6)
    ctx.run()
    self.assertEqual(list(ctx.relation("born_in")), [(0.8, (0, "china")), (0.6, (1, "china"))])

  def test_proofs_disjunction(self):
    ctx = scallopy.ScallopContext(provenance="proofs")
    ctx.add_relation("assignment", (str, bool))
    ctx.add_relation("bf_var", (int, str))
    ctx.add_relation("bf_not", (int, int))
    ctx.add_relation("bf_and", (int, int, int))
    ctx.add_relation("bf_or", (int, int, int))
    ctx.add_relation("bf_root", (int,)) # Note the type here is a tuple (int,), therefore the facts should contain one-tuples
    ctx.add_rule("eval_bf(bf, res) :- bf_var(bf, v), assignment(v, res)")
    ctx.add_rule("eval_bf(bf, !res) :- bf_not(bf, cbf), eval_bf(cbf, res)")
    ctx.add_rule("eval_bf(bf, r1 && r2) :- bf_and(bf, bf1, bf2), eval_bf(bf1, r1), eval_bf(bf2, r2)")
    ctx.add_rule("eval_bf(bf, r1 || r2) :- bf_or(bf, bf1, bf2), eval_bf(bf1, r1), eval_bf(bf2, r2)")
    ctx.add_rule("result(r) :- bf_root(bf), eval_bf(bf, r)")
    ctx.add_facts("bf_var", [(0, "A"), (1, "B")])
    ctx.add_facts("bf_not", [(2, 0), (3, 1)])
    ctx.add_facts("bf_and", [(4, 0, 2), (5, 1, 3)])
    ctx.add_facts("bf_or", [(6, 4, 5)])
    ctx.add_facts("bf_root", [(6,)]) # Note that here (6,) is a one-tuple
    ctx.add_facts("assignment", [("A", False), ("A", True), ("B", False), ("B", True)], disjunctions=[[0, 1], [2, 3]])
    ctx.run()
    self.assertEqual(list(ctx.relation("result")), [(False,)])

  def test_top_k_proofs_disjunction(self):
    ctx = scallopy.ScallopContext(provenance="topkproofs")
    ctx.add_relation("assignment", (str, bool))
    ctx.add_relation("bf_var", (int, str), non_probabilistic=True)
    ctx.add_relation("bf_not", (int, int), non_probabilistic=True)
    ctx.add_relation("bf_and", (int, int, int), non_probabilistic=True)
    ctx.add_relation("bf_or", (int, int, int), non_probabilistic=True)
    ctx.add_relation("bf_root", int, non_probabilistic=True) # Note that the type here is a single type int, therefore the facts could be either one-tuples or single integer
    ctx.add_rule("eval_bf(bf, res) :- bf_var(bf, v), assignment(v, res)")
    ctx.add_rule("eval_bf(bf, !res) :- bf_not(bf, cbf), eval_bf(cbf, res)")
    ctx.add_rule("eval_bf(bf, r1 && r2) :- bf_and(bf, bf1, bf2), eval_bf(bf1, r1), eval_bf(bf2, r2)")
    ctx.add_rule("eval_bf(bf, r1 || r2) :- bf_or(bf, bf1, bf2), eval_bf(bf1, r1), eval_bf(bf2, r2)")
    ctx.add_rule("result(r) :- bf_root(bf), eval_bf(bf, r)")
    ctx.add_facts("bf_var", [(0, "A"), (1, "B")])
    ctx.add_facts("bf_not", [(2, 0), (3, 1)])
    ctx.add_facts("bf_and", [(4, 0, 2), (5, 1, 3)])
    ctx.add_facts("bf_or", [(6, 4, 5)])
    ctx.add_facts("bf_root", [6]) # Note that the fact here is a single integer
    ctx.add_facts("assignment", [(1.0, ("A", False)), (1.0, ("A", True)), (1.0, ("B", False)), (1.0, ("B", True))], disjunctions=[[0, 1], [2, 3]])
    ctx.run()
    self.assertEqual(list(ctx.relation("result")), [(1.0, (False,))])


if __name__ == "__main__":
  unittest.main()
