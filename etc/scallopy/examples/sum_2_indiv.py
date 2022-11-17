import scallopy
import torch

ctx = scallopy.ScallopContext(provenance="difftopkproofsindiv", k=3)
ctx.add_relation("digit_a", int, input_mapping=list(range(10)))
ctx.add_relation("digit_b", int, input_mapping=list(range(10)))
ctx.add_rule("sum_2(a + b) = digit_a(a), digit_b(b)")

# Forward function
compute_sum_2 = ctx.forward_function("sum_2", list(range(19)))

# The tensor of two digits
digit_a = torch.softmax(torch.randn((2, 10)), dim=1)
digit_b = torch.softmax(torch.randn((2, 10)), dim=1)

# Get the result
result = compute_sum_2(digit_a=digit_a, digit_b=digit_b)

print(result)
