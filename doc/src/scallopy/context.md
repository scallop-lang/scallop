# Scallop Context

The most fundamental point of interaction of `scallopy` is `ScallopContext`.
The following is a very simple example setting up a `ScallopContext` to compute the `edge-path` program:

``` py
import scallopy

# Creating a new context
ctx = scallopy.ScallopContext()

# Add relation of `edge`
ctx.add_relation("edge", (int, int))
ctx.add_facts("edge", [(0, 1), (1, 2)])

# Add rule of `path`
ctx.add_rule("path(a, c) = edge(a, c) or path(a, b) and edge(b, c)")

# Run!
ctx.run()

# Check the result!
print(list(ctx.relation("path"))) # [(0, 1), (0, 2), (1, 2)]
```

Roughly, the program above can be divided into three phases:

1. Setup the context: this involves defining relations, adding facts to relations, and adding rules that do the computation
2. Running the program inside of context
3. Fetch the results

While the 2nd and 3rd steps are the place where the computation really happens, it's more important for the programmers to correctly setup the full context for computation.
We now elaborate on what are the high-level things to do when setting up the context

## Configurations

When creating a new `ScallopContext`, one should configure it with intended provenance.
If no argument is supplied, as shown in the above example, the context will be initialized with the default provenance, `unit`, which resembles untagged semantics (a.k.a. discrete Datalog).
To explicitly specify this, you can do

``` py
ctx = scallopy.ScallopContext(provenance="unit")
```

Of course, Scallop can be used to perform reasoning on probabilistic and differentiable inputs.
For instance, you can write the following

``` py
ctx = scallopy.ScallopContext(provenance="minmaxprob") # Probabilistic
# or
ctx = scallopy.ScallopContext(provenance="diffminmaxprob") # Differentiable
```

For more information on possible provenance information, please refer to the [provenance](scallopy/provenance.md) section.
It it worth noting that some provenance, such as `topkproofs`, accept additional parameters such as `k`.
In this case, you can supply this as additional arguments when creating the context:

``` py
ctx = scallopy.ScallopContext(provenance="topkproofs", k=5) # top-k-proofs provenance with k = 5
```

## Adding Program

Given that a context has been configured and initialized, we can set it up the quickest by loading a program into the context.
One can either load an external `.scl` file, or directly inserting a program written as Python string.
To directly add a full program string to the context, one can do

``` py
ctx.add_program("""
  rel edge = {(0, 1), (1, 2)}
  rel path(a, c) = edge(a, c) or path(a, b) and edge(b, c)
""")
```

On the other hand, assuming that there is a file `edge_path.scl` that contains the same content as the above string, one can do

``` py
ctx.import_file("edge_path.scl")
```

## Adding Relations

Instead of adding program as a whole, one can also add relations one-at-a-time.
When adding new relations, one would need to supply the name as well as the type of the relation.
For example, the `edge` relation can be defined as follows

``` py
ctx.add_relation("edge", (int, int))
```

Here, we are saying that `edge` is an arity-2 relation storing pairs of integers.
Note that we are specifying the type using Python's `int` type.
This is equivalent to the `i32` type inside Scallop.
Therefore, the above instruction tranlates to the following Scallop code:

``` scl
rel edge(i32, i32)
```

Many existing Python types can directly translate to Scallop type.
In particular, we have the mapping listed as follows:

| Python Type | Scallop Type |
|-------------|--------------|
| `int` | `i32` |
| `bool` | `bool` |
| `float` | `f32` |
| `str` | `String` |

In case one want to use types other than the listed ones (e.g., `usize`), they can be accessed directly using the string `"usize"`, or they can be accessed through predefined types such as `scallopy.usize`.
The example below defines a relation of type `(usize, f64, i32)`:

``` py
ctx.add_relation("my_relation", (scallopy.usize, "f64", int))
```

Specifically for arity-1 relations, users don't need to use a tuple to specify the type.
For instance,

``` py
ctx.add_relation("digit", int)
```

## Adding Facts

The most basic version of adding facts into an existing relation inside of an existing context.
We are assuming that the context has a provenance of `"unit"`.

``` py
ctx.add_facts("edge", [(1, 2), (2, 3)])
```

If the relation is declared to be having arity-1 and that the type is a singleton type instead of a 1-tuple, then the facts inside of the list do not need to be a tuple.

``` py
ctx.add_relation("digit", int)
ctx.add_facts("digit", [1, 2, 3])
```

### Probabilistic Facts (Tagged Facts)

When the Scallop context is configured to use a provenance other than.
If one wants to add facts along with probabilities, they can wrap their non-probabilistic facts into tuples whose first element is a simple probability.
For example, if originally we have a fact `1`, wrapping it with a corresponding probability gives us `(0.1, 1)`, where `0.1` is the probability.

``` py
ctx.add_facts("digit", [1, 2, 3])                      # without probability
ctx.add_facts("digit", [(0.1, 1), (0.2, 2), (0.7, 3)]) # with probability
```

Of course, if the original facts are tuples, the ones with probability will be required to wrap further:

``` py
ctx.add_facts("color", [("A", "blue"), ("A", "green"), ...])               # without probability
ctx.add_facts("color", [(0.1, ("A", "blue")), (0.2, ("A", "green")), ...]) # with probability
```

We can extend this syntax into tagged facts in general.
Suppose we are using the boolean semiring (`boolean`), we are going to tag each fact using values such as `True` or `False`.

``` py
ctx = scallopy.Context(provenance="boolean")
ctx.add_relation("edge", (int, int))
ctx.add_facts("edge", [(True, (1, 2)), (False, (2, 3))])
```

### Non-tagged Facts in Tagged Context

## Adding Rules

### Tagged Rules

## Running

## Additional Features

There are more features provided by the `ScallopContext` interface.
We hereby list them for reference.

### Cloning

One can copy a context to create a new context.
The resulting context will contain all the program, configurations, and provenance information.

``` py
new_ctx = ctx.clone()
```

The cloning feature relates to pseudo-incremental computation and branching computation.
We elaborate on this in the [Branching Computation](scallopy/branching.md) section.

### Compiling

### Iteration Count Limit

One can configure the

### Early Discarding

### Obtaining Context Information

### Foreign Functions and Predicates

### Saving and Loading

Please refer to the [Saving and Loading](scallopy/save_and_load.md) section for more information.
