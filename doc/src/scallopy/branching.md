# Branching Executions

One cool feature that `scallopy` supports is *branching execution*.
People can create a context, clone it to form new contexts, and modify the new context without touching the old ones.
This is particularly useful when incremental computation is desired.

``` py
# Create the first version of the context
ctx = scallopy.ScallopContext()
ctx.add_relation(...)
ctx.add_facts(...)

# Branch it into another context
ctx1 = ctx.clone()
ctx1.add_relation(...)
ctx1.add_facts(...)
ctx1.run() # Running the first context

# Branch it into one more context; `ctx1` and `ctx2` are completely disjoint
ctx2 = ctx.clone()
ctx2.add_relation(...)
ctx2.add_facts(...)
ctx2.run() # Running the second context
```
