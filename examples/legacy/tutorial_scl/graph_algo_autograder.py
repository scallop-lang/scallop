import os
import scallopy

def test_relation(ctx, relation_name, expected_output):
  result = set(ctx.relation(relation_name))
  expected = set(expected_output)
  in_result_not_expected = result.difference(expected)
  if len(in_result_not_expected) > 0:
    e = in_result_not_expected.pop()
    print(f"❌ Test Failed on relation {relation_name}: unexpected element {e} found in result")
    return
  in_expected_not_result = expected.difference(result)
  if len(in_expected_not_result) > 0:
    e = in_expected_not_result.pop()
    print(f"❌ Test Failed on relation {relation_name}: expected element {e} not found in result")
    return
  print(f"✅ Test Passed on relation {relation_name}")

def test_1():
  print("Autograding Graph Test 1")
  ctx = scallopy.ScallopContext()
  ctx.import_file(os.path.abspath(os.path.join(__file__, "../graph.scl")))
  ctx.add_facts("node", [(0,), (1,), (2,), (3,), (4,), (5,)])
  ctx.add_facts("edge", [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4)])
  ctx.run()
  test_relation(ctx, "path", [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (3, 4)])
  test_relation(ctx, "triangle", [(0, 1, 2), (1, 2, 0), (2, 0, 1)])
  test_relation(ctx, "scc", [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 3), (4, 4), (5, 5)])
  test_relation(ctx, "singleton_scc", [(3,), (4,), (5,)])
  test_relation(ctx, "contains_cycle", [(True,)])
  test_relation(ctx, "num_nodes", [(6,)])
  test_relation(ctx, "in_degree", [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 0)])
  test_relation(ctx, "out_degree", [(0, 2), (1, 1), (2, 1), (3, 1), (4, 0), (5, 0)])
  test_relation(ctx, "num_nodes_within_3", [(0, 5), (1, 4), (2, 5), (3, 1), (4, 0), (5, 0)])
  test_relation(ctx, "shortest_path_length", [(0, 0, 3), (0, 1, 1), (0, 2, 2), (0, 3, 1), (0, 4, 2), (1, 0, 2), (1, 1, 3), (1, 2, 1), (1, 3, 3), (1, 4, 4), (2, 0, 1), (2, 1, 2), (2, 2, 3), (2, 3, 2), (2, 4, 3), (3, 4, 1)])

test_1()
