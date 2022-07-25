import scallopy
import torch

ctx = scallopy.ScallopContext(provenance="difftopbottomkclauses")

ctx.add_relation("obj_color", (int, str))
ctx.add_facts("obj_color", [(torch.tensor(0.99), (0, "blue")), (torch.tensor(0.01), (0, "green"))])
ctx.add_facts("obj_color", [(torch.tensor(0.86), (1, "blue")), (torch.tensor(0.14), (1, "green"))])
ctx.add_facts("obj_color", [(torch.tensor(0.01), (2, "blue")), (torch.tensor(0.99), (2, "green"))])

ctx.add_rule('num_blue_obj(x) :- x = count(o: obj_color(o, "blue"))')

ctx.run()

print(list(ctx.relation("num_blue_obj")))
