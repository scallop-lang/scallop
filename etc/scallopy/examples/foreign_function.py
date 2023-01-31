import scallopy

# Create a generic type parameter
T = scallopy.ScallopGenericTypeParam(scallopy.Number)

# Create a foreign scallop function
# Here the function `my_sum` can accept arbitrary amount of `Number`s
@scallopy.foreign_function
def my_sum(*a: T) -> T:
  s = 0
  for x in a:
    s += x
  return s

# Create a context
ctx = scallopy.ScallopContext()

# Register a foreign function (my_sum)
# Note that the function needs to be registered before it being used
ctx.register_foreign_function(my_sum)

# Add some relation...
ctx.add_relation("R", (int, int))
ctx.add_facts("R", [(1, 2), (2, 3)])

# Add a rule which uses our registered function!
ctx.add_rule("S($my_sum(a, b)) :- R(a, b)")

# Run the context
ctx.run()

# See the result, should be [(3,), (5,)] according to the facts we inserted
print(list(ctx.relation("S")))
