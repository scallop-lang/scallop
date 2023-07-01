# Foreign Functions

While there are existing [foreign functions](../language/foreign_functions.md) such as `$hash` and `$abs`, people sometimes want more functions to be included for specialized computation.
`scallopy` provides such interface and allows user to define foreign functions in Python.
Here is an example defining a custom `$sum` function in Python which is later used in Scallop:

``` py
# Create a new foreign function by annotating an existing function with `@scallopy.foreign_function`
# Note that this function has variable arguments!
@scallopy.foreign_function
def my_sum(*args: int) -> int:
  s = 0
  for x in args:
    s += x
  return s

# Create a context
ctx = scallopy.ScallopContext()

# Register the declared foreign function (`my_sum`)
# Note that the function needs to be registered before it is used
ctx.register_foreign_function(my_sum)

# Add some relations
ctx.add_relation("I", (int, int))
ctx.add_facts("I", [(1, 2), (2, 3), (3, 4)])

# Add a rule which uses the registered function!
ctx.add_rule("R($my_sum(a, b)) = I(a, b)")

# Run the context
ctx.run()

# See the result, should be [(3,), (5,), (7,)]
print(list(ctx.relation("R")))
```

Now we elaborate on how we define new foreign functions in Python.

## Function Signature

The annotator `@scallopy.foreign_function` performs analysis of the annotated Python function and makes sure that it is accepted as a Scallop foreign function.
We require that types are annotated on all arguments and the return value.
For simplicity, Python types such as `int`, `bool`, and `str` are mapped to Scallop types (and type families) as following:

| Python type | Scallop type | Scallop base types |
|-------------|--------------|--------------------|
| `int` | `Integer` family | `i8`, `i16`, ..., `u8`, `u16`, ..., `usize` |
| `float` | `Float` family | `f32`, `f64` |
| `bool` | `bool` | `bool` |
| `str` | `String` | `String` |

If one desires to use a more fine-grained type

## Argument Types

## Optional Arguments

## Variable Arguments

## Error Handling
