import os
import torch
import scallopy

this_file_path = os.path.abspath(os.path.join(__file__, "../"))

# Create scallop context
ctx = scallopy.ScallopContext(provenance="difftopbottomkclauses")
ctx.import_file(os.path.join(this_file_path, "../scl/hwf_parser.scl"))

# The symbols facts
ctx.add_facts("symbol", [
  (torch.tensor(0.2), (0, "3")), (torch.tensor(0.5), (0, "5")),
  (torch.tensor(0.1), (1, "*")), (torch.tensor(0.3), (1, "/")),
  (torch.tensor(0.01), (2, "4")), (torch.tensor(0.8), (2, "2")),
])

# The length facts
ctx.add_facts("length", [
  (None, (3,))
])

# Run the context
ctx.run(debug_input_provenance=True)

# Inspect the result
print(list(ctx.relation("result")))
