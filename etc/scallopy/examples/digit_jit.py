import scallopy
import torch

ctx = scallopy.ScallopContext(provenance="diffminmaxprob")
ctx.add_relation("digit_1", int, list(range(10)))
ctx.add_relation("digit_2", int, list(range(10)))
ctx.add_rule("sum_2(a + b) = digit_1(a), digit_2(b)")
sum_2 = ctx.forward_function("sum_2", list(range(19)), jit=True) # Note that we have JIT enabled

digit_1 = torch.randn((16, 10))
digit_2 = torch.randn((16, 10))
result = sum_2(digit_1=digit_1, digit_2=digit_2)

print(result)
