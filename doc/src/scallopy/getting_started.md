# Getting Started with Scallopy

## Motivating Example

Let's start with a very simple example illustrating the usage of `scallopy`.

``` python
import scallopy

ctx = scallopy.Context()

ctx.add_relation("edge", (int, int))
ctx.add_facts("edge", [(1, 2), (2, 3)])

ctx.add_rule("path(a, c) = edge(a, c) or path(a, b) and edge(b, c)")

ctx.run()

print(list(ctx.relation("path"))) # [(1, 2), (1, 3), (2, 3)]
```

In this very simple edge-path example, we are interacting with Scallop through a Python class called `Context`.
Basically, a `Context` manages a Scallop program, along with the relations, facts, and execution results corresponding to that Scallop program.
We create a `Context` by `ctx = scallopy.Context`.
Relations, facts, and rules are added through the functions `add_relation(...)`, `add_facts(...)`, and `add_rule(...)`.
With everything set, we can execute the program inside the context by calling `run()`
Lastly, we pull the result from `ctx` by using `relation(...)`.
Please refer to a more detailed explanation of this example in [Scallop Context](context.md).

## Machine Learning with Scallopy and PyTorch

When doing machine learning, we usually want to have batched inputs and outputs.
Instead of building the Scallop context incrementally and explicitly run the program, we can create a `Module` at once and be able to run the program for a batch of inputs.
This offers a few advantages, such as optimization during compilation, batched execution for integration with machine learning pipelines, simplified interaction between data structures, and so on.
For example, we can create a module and run it like the following:

``` python
import scallopy
import torch

# Creating a module for execution
my_sum2 = scallopy.Module(
  program="""
    type digit_1(a: i32), digit_2(b: i32)
    rel sum_2(a + b) = digit_1(a) and digit_2(b)
  """,
  input_mappings={"digit_1": range(10), "digit_2": range(10)},
  output_mappings={"sum_2": range(19)},
  provenance="difftopkproofs")

# Invoking the module with torch tensors. `result` is a tensor of 16 x 19
result = my_sum2(
  digit_1=torch.softmax(torch.randn(16, 10), dim=0),
  digit_2=torch.softmax(torch.randn(16, 10), dim=0))
```

As can be seen in this example, we have defined a `Module` which can be treated also as a PyTorch module.
Similar to other PyTorch modules, it can take in torch tensors and return torch tensors.
The logical symbols (such as the `i32` numbers used in `digit_1` and `digit_2`) are configured in `input_mappings` and `output_mappings`, and can be automatically converted from tensors.
We also see that it is capable of handling a batch of inputs (here, the batch size is 16).
Internally, Scallop also knows to execute in parallel, making it performing much faster than normal.
Please refer to [Scallop Module](module.md) for more information.
