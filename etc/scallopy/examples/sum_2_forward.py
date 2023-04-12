import scallopy
import torch

sum_2_program = """
type digit_a(usize), digit_b(usize)
rel sum_2(a + b) = digit_a(a), digit_b(b)
"""

compute_sum_2 = scallopy.ScallopForwardFunction(
  program=sum_2_program,
  provenance="diffaddmultprob2",
  input_mappings={"digit_a": list(range(10)), "digit_b": list(range(10))},
  output_mappings={"sum_2": list(range(19))})

digit_a = torch.softmax(torch.randn((16, 10)), dim=1)
digit_b = torch.softmax(torch.randn((16, 10)), dim=1)
result = compute_sum_2(digit_a=digit_a, digit_b=digit_b)

print(result)
