from typing import List
import os
import random
import functools
import json
from argparse import ArgumentParser
from tqdm import tqdm

def precedence(operator: str) -> int:
  if operator == "+" or operator == "-": return 2
  elif operator == "*" or operator == "/": return 1
  else: raise Exception(f"Unknown operator {operator}")

class Expression:
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data/HWF/Handwritten_Math_Symbols"))

  def sample_images(self) -> List[str]: pass

  def __str__(self) -> str: pass

  def __len__(self) -> int: pass

  def value(self) -> int: pass

  def precedence(self) -> int: pass


class Constant(Expression):
  def __init__(self, digit: int):
    super(Constant, self).__init__()
    self.digit = digit

  def __str__(self):
    return f"{self.digit}"

  def __len__(self):
    return 1

  def sample_images(self) -> List[str]:
    imgs = Constant.images_of_digit(self.digit)
    return [imgs[random.randint(0, len(imgs) - 1)]]

  def value(self) -> int:
    return self.digit

  def precedence(self) -> int:
    return 0

  @functools.lru_cache
  def images_of_digit(digit: int) -> List[str]:
    return [f"{digit}/{f}" for f in os.listdir(os.path.join(Expression.data_root, str(digit)))]


class BinaryOperation(Expression):
  def __init__(self, operator: str, lhs: Expression, rhs: Expression):
    self.operator = operator
    self.lhs = lhs
    self.rhs = rhs

  def sample_images(self) -> List[str]:
    imgs = BinaryOperation.images_of_symbol(self.operator)
    s = [imgs[random.randint(0, len(imgs) - 1)]]
    l = self.lhs.sample_images()
    r = self.rhs.sample_images()
    return l + s + r

  def value(self) -> str:
    if self.operator == "+": return self.lhs.value() + self.rhs.value()
    elif self.operator == "-": return self.lhs.value() - self.rhs.value()
    elif self.operator == "*": return self.lhs.value() * self.rhs.value()
    elif self.operator == "/": return self.lhs.value() / self.rhs.value()
    else: raise Exception(f"Unknown operator {self.operator}")

  def __str__(self):
    return f"{self.lhs} {self.operator} {self.rhs}"

  def __len__(self):
    return len(self.lhs) + 1 + len(self.rhs)

  def precedence(self) -> int:
    return precedence(self.operator)

  @functools.lru_cache
  def images_of_symbol(symbol: str) -> List[str]:
    if symbol == "+": d = "+"
    elif symbol == "-": d = "-"
    elif symbol == "*": d = "times"
    elif symbol == "/": d = "div"
    else: raise Exception(f"Unknown symbol {symbol}")
    return [f"{d}/{f}" for f in os.listdir(os.path.join(Expression.data_root, d))]


class ExpressionGenerator:
  def __init__(self, const_perc, max_depth, max_length, digits, operators, length):
    self.const_perc = const_perc
    self.max_depth = max_depth
    self.max_length = max_length
    self.digits = digits
    self.operators = operators
    self.length = length

  def generate_expr(self, depth=0):
    if depth >= self.max_depth or random.random() < self.const_perc:
      digit = self.digits[random.randint(0, len(self.digits) - 1)]
      expr = Constant(digit)
    else:
      symbol = self.operators[random.randint(0, len(self.operators) - 1)]
      lhs = self.generate_expr(depth + 1)
      if lhs is None or precedence(symbol) < lhs.precedence(): return None
      rhs = self.generate_expr(depth + 1)
      if rhs is None or precedence(symbol) < rhs.precedence(): return None
      if symbol == "/" and rhs.value() == 0: return None
      expr = BinaryOperation(symbol, lhs, rhs)
    if len(expr) > self.max_length: return None
    if depth == 0 and self.length is not None and len(expr) != self.length: return None
    return expr

  def generate_datapoint(self, id):
    while True:
      e = self.generate_expr()
      if e is not None:
        return {"id": str(id), "img_paths": e.sample_images(), "expr": str(e), "res": e.value()}


if __name__ == "__main__":
  parser = ArgumentParser("hwf/datagen")
  parser.add_argument("--operators", action="store", default=["+", "-", "*", "/"], nargs="*")
  parser.add_argument("--digits", action="store", type=int, default=list(range(10)), nargs="*")
  parser.add_argument("--num-datapoints", type=int, default=100000)
  parser.add_argument("--max-depth", type=int, default=3)
  parser.add_argument("--max-length", type=int, default=7)
  parser.add_argument("--length", type=int)
  parser.add_argument("--constant-percentage", type=float, default=0.1)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--output", type=str, default="dataset.json")
  args = parser.parse_args()

  # Parameters
  random.seed(args.seed)
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data/HWF"))

  # Generate datapoints
  generator = ExpressionGenerator(args.constant_percentage, args.max_depth, args.max_length, args.digits, args.operators, args.length)
  data = [generator.generate_datapoint(i) for i in tqdm(range(args.num_datapoints))]

  # Dump data
  json.dump(data, open(os.path.join(data_root, args.output), "w"))
