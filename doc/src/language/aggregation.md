# Aggregations

Aggregations in Scallop can be viewed as operations that aggregates over multiple facts.
Such operations include counting, summation and product, finding min and max, and logical quantifiers such as exists and forall.
Aggregations appear in the body of a rule, and can be nested for abbrevity.

As a concrete example, we look at a program which counts over a set of people:

``` scl
rel person = {"alice", "bob", "christine"}
rel num_people(n) = n := count(p: person(p)) // n = 3
```

<!-- While further down the road we are going to discuss their probabilistic (multi-world) semantics, let's first view them as discrete logical operations. -->
In general, we use the following syntax for aggregation formulas.

``` scl
R1, R2, ... := AGGREGATOR(V1, V2, ...: FORMULA (where U1, U2, ...: FORMULA)?)
```

We name `R1, ...` to be the aggregation *result* variable, `V1, ...` to be the *binding* variable, and the formula inside of the aggregation the *body*.
When the `where` keyword is used, we have the aggregation associated with *explicit group-by* clause.
Here, we call the set of variables `U1, ...` as *group-by variables*.
The formula under the `where` clause is named the *group-by body*.
The binding variables need to be fully grounded by the body formula, and the group-by variables (if presented) need to also be fully grounded by the group-by body.
For different types of aggregation, the `AGGREGATOR` might also change and annotated with different information.
The number of result variables, the number of binding variables, and their types differ for each aggregation.

Here is a high-level overview of each supported aggregator and their configurations.
In the table, `...` is used to denote an arbitrary amount of variables.

| Aggregator | Binding Variables | Result Variables |
|------------|-------------------|------------------|
| `count` | `Any...` | `usize` |
| `sum` | `Number` | the same as the binding variable |
| `prod` | `Number` | the same as the binding variable |
| `min` | `Any` | the same as the binding variables |
| `max` | `Any` | the same as the binding variables |
| `exists` | `Any...` | `bool` |
| `forall` | `Any...` | `bool` |

Below, we elaborate on each aggregators and describe their usages.

## Count

To count the number of facts, we can use the `count` aggregator.
Just repeating the examples shown in the beginning:

``` scl
rel person = {"alice", "bob", "christine"}
rel num_people(n) = n := count(p: person(p)) // n = 3
```

We are counting the number of persons appear in the `person` relation.
To be more concrete, let's read out the aggregation formula:

> We count the number of `p` such that `p` is a `person`, and assign the result to the variable `n`.

For `count`, there could be arbitrary (> 0) number of binding variables which can be typed arbitrarily.
It will only have a single result variable which is typed `usize`.
For example, you may count the number of `edge`s:

``` scl
rel num_edges(n) = n := count(a, b: edge(a, b))
```

Here, we have two binding variables `a` and `b`, meaning that we are counting the number of *distinct* pairs of `a` and `b`.

Note that we can use the syntax sugar for aggregation to omit the repeated `n`:

``` scl
rel num_edges = count(a, b: edge(a, b))
```

### Implicit Group-By

With `group-by`, we may count the number of facts under a pre-defined group.
Consider the example where there is a scene with differet colored objects,

``` scl
rel obj_color = {(0, "red"), (1, "red"), (2, "blue"), (3, "red")}
rel num_obj_per_color(col, num) = num := count(obj: obj_color(obj, col))
```

As suggested by the facts inside of `obj_color`, there are `4` objects indexed using `0, 1, 2, 3`, each associated with a different color.
The object #0, #1, and #3 are `red` and the object #2 is `blue`.
Therefore, we will get 3 red objects and 1 blue object, as computed in the result of `num_obj_per_color`:

```
num_obj_per_color: {("blue", 1), ("red", 3)}
```

Let's analyze the rule in detail.
We find that we are counting over `obj` such that the object `obj` has a certain color `col`.
But `col` is also a variable occurring in the head of the rule.
This is an *implicit group-by*, in that the variable `col` is being used as an implicit group-by variable.
That is, we are conditioning the counting procedure under each *group* that is defined by the `col` variable.
Since there are two colors appearing in the `obj_color` relation, we are performing count for each of the two groups.

In general, if a variable is positively grounded in the body and appear in the head of a parent rule, we call the variable an *implicit group-by variable*.

### Explicit Group-By

In the above example, there is no green colored object.
However, how do we know that the number of green object is 0?
The result does not seem to address this problem.

The missing piece is a *domain* of the possible groups.
Without explicitly setting the domain, Scallop could only search inside of the database on possible groups.
However, we can explicitly tell Scallop about what are the groups.
Consider the following rewrite of the above program:

``` scl
rel colors = {"red", "green", "blue"}
rel obj_color = {(0, "red"), (1, "red"), (2, "blue"), (3, "red")}
rel num_obj_per_color(col, num) = num := count(obj: obj_color(obj, col) where col: colors(col))
```

With the `where` clause, we have explicitly declared that `col` is a *group-by variable* which is grounded by the `colors` relation.
If we look into the `colors` relation, we find that there are three possible colors that we care about, red, green, and blue.
In this case, we will consider `"green"` as the third group and try to count the number of green objects -- which there are 0:

```
num_obj_per_color: {("blue", 1), ("green", 0), ("red", 3)}
```

## Sum and Product

We can use the aggregator of sum and product to aggregate multiple numerical values.
Consider the following example of sales:

``` scl
rel sales = {("alice", 1000.0), ("bob", 1200.0), ("christine", 1000.0)}
```

We can compute the sum of all the sales:

``` scl
rel total_sales(s) = s := sum[p](sp: sales(p, sp)) // 3200.0
// or
rel total_sales = sum[p](sp: sales(p, sp)) // 3200.0
```

Notice that the result type of `s` is the same as the type of the binding variable `sp`, which is `f32` as indicated by the decimals in the definition of `sales`.
Here, the argument variable `p` is necessary since it is the key to index each sale number.
The above rule body is equivalent to the following math formula:

\\[ s = \sum_p \text{sale}_p \\]

If we do not use the argument variable, we get the following:

``` scl
rel total_sales_wrong(s) = s := sum(sp: sales(p, sp)) // 2200.0, since the two 1000.0 will be deduplicated without its key
```

The product aggregator `prod` can be used in a similar manner as `sum`.

## Min, Max, Argmin, and Argmax

Scallop can compute the minimum or maximum among a set of values.
In the following example, we find the maximum grade of an exam:

``` scl
rel exam_grades = {("a", 95.2), ("b", 87.3), ("c", 99.9)}
rel min_score(m) = m := max(s: exam_grades(_, s)) // 99.9
// or, succinctly
rel min_score = max(s: exam_grades(_, s)) // 99.9
```

The number (and types) of binding variables can be arbitrary, but the result variables must match the binding variables.
In the above case, since `s` is of type `f32`, `m` will be of type `f32` as well.

It is also possible to get argmax/argmin.
Suppose we want to get the person (along with their grade) who scored the best, we write:

``` scl
rel best_student(n, s) = (n, s) := max[n](s: exam_grades(n, s))
// or, succinctly
rel best_student = max[n](s: exam_grades(n, s))
```

Here, we are still finding the maximum score `s`, but along with `max` we have specified the "arg" (`[n]`) which associates with the maximum score.
We call `n` an arg variable for `min`/`max` aggregator.
The arg variable is grounded by the aggregation body, and can be directly used in the head of the rule.

If we do not care about the grade and just want to know who has the best grade, we can use wildcard `_` to ignore the result variable, like

``` scl
rel best_student(n) = (n, _) := max[n](s: exam_grades(n, s))
```

Alternatively, we can also use `argmax`:

``` scl
rel best_student(n) = n := argmax[n](s: exam_grades(n, s))
// or, succinctly
rel best_student = argmax[n](s: exam_grades(n, s))
```

## Exists and Forall

Logical quantifier such as exists and forall can also be encoded as aggregations.
They will return value of boolean as the aggregation result.

### Existential Quantifier

Let us start with discussing the easier of the two, `exists`.
Technically, all variables in the body of Scallop rule are existentially quantified.
We can use `exists` aggregation to make it explicit.
For example, we can check if there exists an object that is blue:

``` scl
rel obj_color = {(0, "red"), (1, "green")}
rel has_blue(b) = b := exists(o: obj_color(o, "blue"))
```

Specifically, we are checking "if there exists an object `o` such that its color is `blue`".
The result is being assigned to a variable `b`.
Since there is no blue object, we will get a result of `has_blue(false)`.

In case when we just want the result boolean to be `true` or `false`, we can omit the result variables.
For example, we can rewrite the recursive case of edge-path transitive closure as

``` scl
rel path(a, c) = exists(b: path(a, b) and edge(b, c))
```

We note that this is just a syntax sugar equivalent to the following:

``` scl
rel path(a, c) = r := exists(b: path(a, b) and edge(b, c)) and r == true
```

When we want to know the inexistence of something, we can do

``` scl
rel no_red() = not exists(o: obj_color(o, "red"))
```

Note that there can be arbitrary amount of binding variables.

### Universal Quantifier

We can also have universal quantifier `forall`.
For this, there is a special requirement for universal quantification: the body formula has to be an `implies` (`=>`) formula.
This restriction is enforced so that all the binding variables have bounds being specified on the left-hand-side of the `implies` formula.
In the following example, we check if all the objects are spherical:

``` scl
type Shape = CUBE | SPHERE | CONE | CYLINDER
rel object = {0, 1, 2}
rel obj_shape = {(0, CUBE), (1, SPHERE), (2, SPHERE)}
rel target(b) = b := forall(o: object(o) implies obj_shape(o, SPHERE))
```

Notice that we have a relation which defines the domain of `object`, suggesting that there are just 3 objects for us to work with.
In the aggregation, we are checking "for all `o` such that `o` is an object, is the object a sphere?"
The result is stored in the variable `b` and propagated to the `target` relation.

The reason we need to have an *implies* formula is that we need to use the left-hand-side of `implies` to give bounds to the universally quantified variables.
Scallop cannot reason about open domain variables.

Note that similar to `exists`, we can also remove the result variable.
The following program derives a boolean (arity-0) relation `target` denoting whether all the red objects are cubes:

``` scl
type Shape = CUBE | SPHERE | CONE | CYLINDER
type Color = RED | GREEN | BLUE
rel obj_shape = {(0, CUBE), (1, SPHERE), (2, SPHERE)}
rel obj_color = {(0, RED),  (1, GREEN),  (2, GREEN)}
rel target() = forall(o: obj_color(o, RED) implies obj_shape(o, CUBE)) // {()}
```

Here, we directly use `obj_color` to serve as the left-hand-side of the `implies`.
There will be one empty tuple being derived, suggesting that the statement is true.

## String Join

If you have multiple facts containing strings and you want to join them together, you can use the `string_join` aggregator:

``` scl
rel R = {"hello", "world"}
rel P1(n) = n := string_join(s: R(s)) // P1("helloworld")
rel P2(n) = n := string_join<" ">(s: R(s)) // P2("hello world")
```

In the above example,
we can either directly join, producing the string "helloworld",
or join with separator `" "`, producing the string "hello world".
Note that the order of the strings in the joined string is determined by the strings.
Here, `"hello"` starts with `"h"`, which is smaller than the `"w"` in `"world"`, therefore occurring before `"world"`.
If you want to specify an explicit order, use the argument variable:

``` scl
rel R = {(2, "hello"), (1, "world")}
rel P(n) = n := string_join<" ">[i](s: R(i, s)) // P("world hello")
```

Since we have specified the variable `i` to be the argument of `string_join`, it serves to order the tuples.
Here, we have `(1, "world")` and `(2, "hello")`, so the joined string will be `"world hello"` instead of `"hello world"`.
