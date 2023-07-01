# Reference Guide

We list all the language features supported by Scallop.

## Import Files

``` scl
import "path/to/other/file.scl"
```

## Type Definition

### Type Alias Definition

``` scl
type ObjectId = usize
```

### Sub-Type Definition

``` scl
type Name <: String
```

### Enum Type Definition

``` scl
type Action = LEFT | RIGHT | UP | DOWN
```

### Algebraic Data Type Definition

``` scl
type Expr = Const(i32) | Add(Expr, Expr) | Sub(Expr, Expr)
```

### Relation Type Definition

``` scl
type edge(x: i32, y: i32)
```

## Constant Definition

``` scl
const PI: f32 = 3.1415
```

## Relation Definition

### Fact Definition

``` scl
rel edge(1, 2)
```

### Set-of-Tuples Definition

``` scl
rel edge = {(1, 2), (2, 3), (3, 4)}
```

### Rule Definition

``` scl
rel path(a, b) = edge(a, b) or path(a, c) and edge(c, b)
```

#### Disjunctive Head

``` scl
rel { assign(v, false); assign(v, true) } = variable(v)
```

#### Atom

``` scl
fib(x - 1, y)
```

#### Negation

``` scl
rel has_no_child(p) = person(p) and not father(p, _) and not mother(p, _)
```

#### Constraint

``` scl
rel number(0)
rel number(i + 1) = number(i) and i < 10
```

#### Aggregation

``` scl
rel person = {"alice", "bob", "christine"}
rel num_people(n) = n := count(p: person(p))
```

#### Foreign Predicate

``` scl
rel grid(x, y) = range<i32>(0, 5, x) and range<i32>(0, 5, y)
```

## Query Definition

``` scl
query path
```
