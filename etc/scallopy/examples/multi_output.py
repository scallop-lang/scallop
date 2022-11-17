import scallopy
import torch

ctx = scallopy.ScallopContext(provenance="diffaddmultprob")
ctx.add_relation("digit_1", int, input_mapping=list(range(10)))
ctx.add_relation("digit_2", int, input_mapping=list(range(10)))
ctx.add_rule("sum_2(a + b) = digit_1(a) and digit_2(b)")
ctx.add_rule("mult_2(a * b) = digit_1(a) and digit_2(b)")
forward = ctx.forward_function(output_mappings={"sum_2": list(range(20)), "mult_2": list(range(100))})

digit_1 = torch.randn((16, 10))
digit_2 = torch.randn((16, 10))
result = forward(digit_1=digit_1, digit_2=digit_2)
print("sum_2 result:", result["sum_2"])
print("mult_2 result:", result["mult_2"])
