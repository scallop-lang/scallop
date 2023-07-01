import unittest
import torch

import scallopy

class TestIO(unittest.TestCase):
  def test_load_csv(self):
    ctx = scallopy.Context()
    ctx.add_relation("edge", (int, int), load_csv=scallopy.io.CSVFileOptions("core/res/testing/csv/edge.csv"))
    ctx.run()
    assert list(ctx.relation("edge")) == [(0, 1), (1, 2), (2, 3)]

  def test_load_csv_with_field(self):
    ctx = scallopy.Context()
    csv_file = scallopy.io.CSVFileOptions("core/res/testing/csv/student.csv", fields=["id", "name", "year"])
    ctx.add_relation("student", (int, str, int), load_csv=csv_file)
    ctx.run()
    assert list(ctx.relation("student")) == [(1, "alice", 2022), (2, "bob", 2023)]

  def test_load_csv_with_key_and_field(self):
    ctx = scallopy.Context()
    csv_file = scallopy.io.CSVFileOptions("core/res/testing/csv/student.csv", keys="id", fields=["name", "year"])
    ctx.add_relation("student", (int, scallopy.Symbol, str), load_csv=csv_file)
    ctx.run()
    assert list(ctx.relation("student")) == [
      (1, "name", "alice"),
      (1, "year", "2022"),
      (2, "name", "bob"),
      (2, "year", "2023"),
    ]
