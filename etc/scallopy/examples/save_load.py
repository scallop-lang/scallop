import os
import torch
import scallopy

class SimpleNet(torch.nn.Module):
  def __init__(self):
    super(SimpleNet, self).__init__()
    self.fc = torch.nn.Linear(8, 8)
    self.ctx = scallopy.ScallopContext(provenance="diffminmaxprob")
    self.ctx.add_relation("inp", int, input_mapping=list(range(8)))
    self.ctx.add_rule("outp(x + 1) = inp(x)")
    self.ev = self.ctx.forward_function("outp", output_mapping=[i + 1 for i in range(8)])

  def forward(self, x):
    return self.ev(inp=self.fc(x))

# Random generate input
x = torch.randn(3, 8)

# Temporary file name
file_name = "tmp.pt"

# First run through source model and get the first result
source_model = SimpleNet()
y1 = source_model(x)

# Then store the model
torch.save(source_model, file_name)

# Remove the original model just to be sure
del source_model

# Load the model, run through it again
loaded_model = torch.load(file_name)
y2 = loaded_model(x)

# Make sure that y1 and y2 are the same
print(y1)
print(y2)

# Remove that temporary file
os.remove(file_name)
