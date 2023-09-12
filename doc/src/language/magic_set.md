# On-Demand Relations

There are often times relations/predicates where you know that would not need to be fully computed.
This would include the infinite relations.
This means that, we want to define such relations without worrying about its infinite-ness while also being able to supply it with information needed for the computation.
Such relations are called **On-Demand Relations**.

We show here one on-demand relation which is the `fibonacci` number relation:

``` scl
type fib(bound x: i32, y: i32)
rel fib = {(0, 1), (1, 1)}
rel fib(x, y1 + y2) = fib(x - 1, y1) and fib(x - 2, y2) and x > 1
query fib(10, y)
```

Normally, if we define the fibonacci relation, it would only contain the second and the third line, which respectively defines the base cases and the recursive cases.
However, as we all know, there are infinitely many fibonacci numbers and it would not be wise to compute the relation fully.
Usually, we want to infer some fact inside of the infinite relation, based on some inputs.
In this case, as noted on the last line, we want to know the 10th fibonacci number.

It is hinted that when we want to compute a fibonacci number, we usually supply the `x` value, in this case, 10, in order to get the value `y`.
This is exactly what we tell the compiler in the first line.
Inside of the type declaration, we provide an additional **adornment** to each of the variables.

- `x` is adorned by `bound`, denoting that it is treated as an **input** (or **bounded**) variable to the relation
- `y` is not adorned by anything, suggesting that it is a **free** variable which will be computed by the rules of the relation

> Getting `x` based on `y` is out-of-scope in this tutorial.

By providing the adornments (with at least one `bound`), we are telling Scallop that the relation should be computed *on-demand*.
From there, Scallop will search for every place where the relation is **demanded**, and restrict the computation of the relation only on the demand.

In our case, there is just one single place where the `fib` relation is demanded (where `x` is `10`).
Therefore, Scallop will compute only the necessary facts in order to derive the final solution.

## Adornments

There are only two kinds of adornments:

- `bound`
- `free`

Annotating whether the variable is treated as *bounded* variable or *free* variable.

If an adornment is not provided on a variable, then it is by default a `free` variable.
In this sense, all normal relations without any adornment would be treated as **non**-on-demand relations.

When at least one `bound` adornment is annotated on a relation type declaration, we know that the relation needs to be computed *on-demand*.

## More Examples

### On-Demand Path

Let's go back to our example of edge-and-path.
Consider that there is a huge graph, but we only want to know a path ending at a specific node:

``` scl
rel path(a, b) = edge(a, b) or (edge(a, c) and path(c, b))
query path(a, 1024)
```

In this case, enumerating all paths would be strictly more expensive than just exploring from the end point.
Therefore, we add an adornment to the `path` relation like the following:

``` scl
type path(free i32, bound i32)
```

We say the second argument is `bound` and the first argument is `free`, matching what we expect from the query.

### On-Demand To-String

Let's consider an simple arithmetic expression language and a `to_string` predicate for the language:

``` scl
type Expr = Const(i32) | Var(String) | Add(Expr, Expr) | Sub(Expr, Expr)

rel to_string(e, $format("{}", i))             = case e is Const(i)
rel to_string(e, $format("{}", v))             = case e is Var(e)
rel to_string(e, $format("({} + {})", s1, s2)) = case e is Add(e1, e2) and to_string(e1, s1) and to_string(e2, s2)
rel to_string(e, $format("({} - {})", s1, s2)) = case e is Sub(e1, e2) and to_string(e1, s1) and to_string(e2, s2)
```

Now that let's say there are many expressions declared as constants:

``` scl
const EXPR_1 = Add(Const(1), Add(Const(5), Const(3)))
const EXPR_2 = Add(Const(1), Var("x"))
const EXPR_3 = Const(13)
```

Scallop would have automatically generated string for all of the expressions.

However, let's say we are only interested in one of the expressions:

``` scl
query to_string(EXPR_3, s)
```

Then most of the computations for `to_string` would be redundant.

In this case, we would also declare `to_string` as an on-demand predicate, like this:

``` scl
type to_string(bound Expr, String)
```

Then only the queried expression will be `to_string`-ed.

## Internals

Internally, when there are relations being annotated with adornments, the whole Scallop program is undergone a program transformation.
This transformation is traditionally called **Magic-Set Transformation** or **Demand Transformation**.
There are multiple papers on the topic, which we reference below:

- [Extended Magic for Negation (Tuncay Tekle et. al. 2019)](https://arxiv.org/pdf/1909.08246.pdf)
- [Precise complexity analysis for efficient datalog queries (Tuncay Tekle et. al. 2010)](https://dl.acm.org/doi/10.1145/1836089.1836094)
- [Efficient bottom-up computation of queries on stratified databases (Balbin et. al. 1991)](https://www.sciencedirect.com/science/article/pii/074310669190030S)
