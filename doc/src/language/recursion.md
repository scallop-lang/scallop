# Recursive Rules

One very powerful programming construct with Scallop is to declaratively define recursion.
Inside of a rule, if a relational predicate appearing in the head appears in the body, the predicate is recursive.
For example, the definition of fibonacci number is recursive:

\\[ \text{fib}(x) = \left\\{ \begin{array}{ll} \text{fib}(x - 1) + \text{fib}(x - 2), & \text{if}~ x > 1 \\\\ 1, & \text{otherwise} \end{array} \right. \\]

Written in Scallop, we encode the function `fib` as a binary relation between the integer input and output:

``` scl
type fib(x: i32, y: i32)
```

We can define the base cases for \\(\text{fib}(0)\\) and \\(\text{fib}(1)\\):

``` scl
rel fib = {(0, 1), (1, 1)}
```

Now it comes to the definition of recursive cases, which peeks into \\(\text{fib}(x - 1)\\) and \\(\text{fib}(x - 2)\\) and sums them.

``` scl
rel fib(x, y1 + y2) = fib(x - 1, y1) and fib(x - 2, y2) // infinite-loop
```

However, when actually executing this, it would not terminate as we are attempting to compute all fibonacci numbers, and there are infinite amount of them.
In order to stop it, we can temporarily add a constraint to limit the value of `x`, so that we only compute the fibonacci number up to 10:

``` scl
rel fib(x, y1 + y2) = fib(x - 1, y1) and fib(x - 2, y2) and x <= 10
```

At the end, we would get a the `fib` relation to contain the following facts:

```
fib: {(0, 1), (1, 1), (2, 2), (3, 3), (4, 5), (5, 8), (6, 13), (7, 21), (8, 34), (9, 55), (10, 89)}
```

As suggested by the result, the 10-th fibonacci number is 89.

## Case Study: Graphs and Transitive Closure

Following is one of the most widely known Datalog program: computing the `path`s inside of a graph.
By definition, an edge or a sequence of edges constitute a path.
This is reflected by the following two rules:

``` scl
type edge(i32, i32)

rel path(a, b) = edge(a, b)
rel path(a, c) = path(a, b) and edge(b, c)
```

The first line states that an edge can form a path.
The second line states that a path, connected to a new edge, forms a new path.
As can be seen from the second line, the relation `path` appears in both the body and the head, making it a *recursive relation*.

In this example, suppose we have

``` scl
rel edge = {(0, 1), (1, 2)}
```

we would get the set of paths to be

```
path: {(0, 1), (0, 2), (1, 2)}
```

Notice that the path `(0, 2)` is a compound path obtained from joining the two edges `(0, 1)` and `(1, 2)`.

## Relation Dependency

Given a rule with head and body, we say that the predicate appearing in the head *depends* on the predicates of the atoms appearing in the body.
This forms a dependency graph.
The above edge-path example would have the following dependency graph:

```
edge <--- path <---+
            |      |
            +------+
```

The relation `edge` depends on nothing, while `path` depends on `edge` and also `path` itself.
This forms a loop in the dependency graph.
In general, if a program has a dependency graph with a loop, then the program requires *recursion*.
Any relation that is involved in a loop would be a *recursive relation*.

Notice that we are mostly talking about *positive dependency* here, as the atoms in the body of the rule are *positive atoms* (i.e., without annotation of negation or aggregation).
In more complex scenarios, there will be negation or aggregation in a rule, which we explain in detail in future sections.

## Fixed-point Iteration

The recursion in Scallop happens in *fixed-point iteration*.
In plain terms, the recursion will continue until there is no new fact being derived in an iteration.
In hind-sight, the whole Scallop program is executed in a loop.
Within one iteration, all of the rules inside of the program are executed.
Let us digest the actual execution happens when executing the above edge-path program:

``` scl
rel edge = {(0, 1), (1, 2), (2, 3)}
rel path(a, b) = edge(a, b)                 // rule 1
rel path(a, c) = path(a, b) and edge(b, c)  // rule 2
```

Before the first iteration, the `edge` has already been filled with 3 facts, namely `(0, 1)`, `(1, 2)`, and `(2, 3)`.
But the `path` is empty.
Let's now go through all the iterations:

```
Iter 0: path = {}
Iter 1: path = {(0, 1), (1, 2), (2, 3)}
       Δpath = {(0, 1), (1, 2), (2, 3)} // through applying rule 1
Iter 2: path = {(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)}
       Δpath = {(0, 2), (1, 3)}         // through applying rule 2
Iter 3: path = {(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)}
       Δpath = {(0, 3)}                 // through applying rule 2
Iter 4: path = {(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)}
       Δpath = {}
```

In the above note, we also include `Δpath`, which contains the new paths derived during the current iteration.
As can be seen, during iteration 1, paths of length 1 are derived; during iteration 2, paths of length 2 are derived.
During iteration 4, there is no more path to be derived, and therefore the `Δpath` is empty.
This tells us that no new facts are derived and the whole fixed-point iteration is stopped, giving us the final result.

## Infinite Relations

As we have described in the *fixed-point iteration*, the recursion will continue until no more fact is derived.
However, we are capable of writing rules that are infinite.
As shown in the first example:

``` scl
rel fib(x, y1 + y2) = fib(x - 1, y1) and fib(x - 2, y2)
```

gives you an infinite relation as there can always be a new `x` to be derived.
In this case, the fixed-point iteration never stops.

The root cause of this is Scallop's support for *value creationg*, i.e., the creation of new values.
Typically, database systems work in closed-world assumption, that is, all the items being reasoned about are already there.
No computation is done on arbitrarily created values.
But in the above example, we have derived `x` from the grounded expression `x - 1`, hence creating a new value.

Typically, the way to resolve this is to create bounds on the created values.
For example, the rule

``` scl
rel fib(x, y1 + y2) = fib(x - 1, y1) and fib(x - 2, y2) and x <= 10
```

restricts that `x` cannot be greater than 10.
This makes the fixed-point iteration to stop after around 10 iterations.

Other way of getting around with this involve the use of [*Magic-Set Transformations*](https://dl.acm.org/doi/pdf/10.1145/6012.15399), which we describe its equivalent in Scallop in [a later section](magic_set.md).
