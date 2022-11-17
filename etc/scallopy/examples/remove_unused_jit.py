import scallopy
import torch

ctx = scallopy.ScallopContext(provenance="diffminmaxprob")
ctx.add_relation("digit_1", int, list(range(10)))
ctx.add_relation("digit_2", int, list(range(10)))
ctx.add_rule("sum_2(a + b) = digit_1(a) and digit_2(b)")
ctx.add_rule("mult_2(a + b) = digit_1(a) and digit_2(b)")
f = ctx.forward_function("sum_2", list(range(19)), jit=True)
result = f(digit_1=torch.randn((16, 10)), digit_2=torch.randn((16, 10)))
print(result)
