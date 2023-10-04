# Algebraic Data Type and Entities

Algebraic data types are powerful programming constructs that allows user to define custom data structures and variants.
Consider a traditional functional definition of a `List`:

``` scl
type IntList = Nil()
             | Cons(i32, List)
```

We are saying that a `IntList` can be one of two variants, `Nil` and `Cons`:
- `Nil` denotes the end of a list;
- `Cons` contains the current `i32` integer and a continuation of the list.

In this representation, we can represent a list like `[1, 2, 3]` with `Cons(1, Cons(2, Cons(3, Nil())))`.
This is indeed what we can write in Scallop.
We can declare such a list as a constant:

``` scl
const MY_LIST = Cons(1, Cons(2, Cons(3, Nil())))
```

In general, we call the type definition of such data structure *Algebraic Data Type* definitions, or *ADT* definitions.
The name *Entity* is used to refer to objects of such data types.
In the example above, the constant `MY_LIST` is an *entity* of the *ADT* named `IntList`.

In this section, we describe in detail the definition and use of ADT and Entities.
We also touch on the internals.

## Defining Algebraic Data Types (ADT)

We use the following syntax to define ADTs:

``` scl
type TYPE_NAME = VARIANT_NAME(ARG_TYPE_1, ARG_TYPE_2, ...) | ...
```

An ADT named `TYPE_NAME` is defined to have multiple (at least 2) named variants with `VARIANT_NAME`.
Each variant holds a tuple of values typed by `ARG_TYPE_1`, `ARG_TYPE_2`, etc.
We call variants that have no argument *terminal variant*s.
Parenthesis are still needed for those variants.

Please note that there cannot be duplicated variant names, either within the same ADT or different ADTs.
For example, the following code would result in compilation failure:

``` scl
type IntList  = Cons(i32, IntList)   | Nil()
type BoolList = Cons(bool, BoolList) | Nil() // Failure: Cons and Nil are already defined
```

Currently, ADTs do not support generics.
In the above case, the `IntList` and `BoolList` needs to be defined separately with differently named variants.

### Using ADT to represent arithmetic expressions

Common data that can be expressed through ADT could be structured expressions.
The following definition describes the abstract syntax tree (AST) of simple arithmetic expressions:

``` scl
type Expr = Int(i32)        // An expression could be a simple integer,
          | Add(Expr, Expr) // a summation of two expressions
          | Sub(Expr, Expr) // a substraction of two expressions
```

The following code encodes a simple expression

``` scl
// The expression (1 + 3) - 5
const MY_EXPR = Sub(Add(Int(1), Int(3)), Int(5))
```

### Using ADT to represent data structures

Data structures such as binary trees can also be represented:

``` scl
type Tree = Node(i32, Tree, Tree) | Nil()
```

Here, `Node(i32, Tree, Tree)` represents a node in a tree holding three things:
an integer (`i32`), a left sub-tree `Tree`, and a right sub-tree `Tree`.
The other variant `Nil` represents an empty sub-tree.
In this encoding, `Node(5, Nil(), Nil())` would be representing a leaf-node holding a number 5.

The following code encodes a balanced binary search tree:

``` scl
//         3
//      /     \
//    1         5
//  /   \     /   \
// 0     2   4     6
const MY_TREE =
  Node(3,
    Node(1,
      Node(0, Nil(), Nil()),
      Node(2, Nil(), Nil()),
    ),
    Node(5,
      Node(4, Nil(), Nil()),
      Node(6, Nil(), Nil()),
    )
  )
```

## Working with Entities

Entities are most commonly created as constants using the `const` keyword.
Let us revisit the `List` example and see how we can use the defined constant in our analysis.

``` scl
type List = Cons(i32, List) | Nil()

const MY_LIST = Cons(1, Cons(2, Cons(3, Nil()))) // [1, 2, 3]
```

### Using Entities in Relations

We can include the constant entities as part of a fact:

``` scl
rel target(MY_LIST)
query target
```

As a result of the above program, we are going to get the value of the entity `MY_LIST`:

```
target: {(entity(0xff08d5d60a201f17))}
```

The value is going to be a 64-bit integer encoded in hex.
It is a unique identifier for the created entity.

Note that, identical entities are going to have the same identifier.
In the following example, `MY_LIST_1` and `MY_LIST_2` are identical, and therefore their hex identifier are the same.

``` scl
const MY_LIST_1 = Cons(1, Nil()),
      MY_LIST_2 = Cons(1, Nil()),
      MY_LIST_3 = Cons(2, Nil())

rel lists = {
  (1, MY_LIST_1),
  (2, MY_LIST_2),
  (3, MY_LIST_3),
}

query lists
// lists: {
//   (1, entity(0x678defa0a65c83ab)), // Notice that the entity 1 and 2 are the same
//   (2, entity(0x678defa0a65c83ab)),
//   (3, entity(0x3734567c3d9f8d3f)), // This one is different than above
// }
```

### Decomposing Entities in Rules

To peek into the content of an Entity, we can *destruct* it using the `case`-`is` operator.
We look at an example of computing the length of a list:

``` scl
type length(list: List, len: i32)
rel length(list, 0)     = case list is Nil()
rel length(list, l + 1) = case list is Cons(_, tl) and length(tl, l)
```

We define a recursive relation `length` to compute the length of a list.
There are two cases.
When the list is `Nil()`, this means the list has already ended.
Therefore the list has a length of `0`
For the second case, the list is `Cons(_, tl)`.
Here, the length of list is the length of `tl` plus 1.

We can then compute the length of a list by `query`ing the `length` relationship on a constant list.

``` scl
query length(MY_LIST, l) // l = 3
```

### Case Study: Decomposing Entities for Pretty-Printing

We can look at more examples of using the `case`-`is` operators.
The following set of rules pretty-prints expressions:

``` scl
type Expr = Int(i32) | Add(Expr, Expr) | Sub(Expr, Expr)

type to_string(expr: Expr, str: String)
rel to_string(e, $format("{}", i))           = case e is Int(i)
rel to_string(e, $format("({} + {})", a, b)) = case e is Add(e1, e2) and to_string(e1, a) and to_string(e2, b)
rel to_string(e, $format("({} - {})", a, b)) = case e is Sub(e1, e2) and to_string(e1, a) and to_string(e2, b)
```

Shown in the example, we have written three `to_string` rules for pretty-printing the `Expr` data structure.
Each rule correspond to handling exactly one of the variants.
For the inductive cases `Add` and `Sub`, we have the `to_string` rule defined recursively so that the sub-expressions are also converted to strings.
For pretty-printing, we have used the `$format` foreign function.

At the end, running the following snippet

``` scl
const MY_EXPR = Sub(Add(Int(3), Int(5)), Int(1))
query to_string(MY_EXPR, s)
```

would give the following result, suggesting that the pretty-printed expression is `((3 + 5) - 1)`

```
to_string(MY_EXPR, s): {(entity(0xa97605c2703c6249), "((3 + 5) - 1)")}
```

### Case Study: Checking Regular Expressions

With ADT, we can specify the language of regular expressions (regex) with ease.
Let's consider a very simple regex with union (`|`) and star (`*`), while phrases can be grouped together.
For example, the regex `"a*b"` expresses that character `a` can be repeated arbitrary amount of time (including 0-times), followed by a single `b`.
This regex can be used to match strings like `"aaaab"` and `"b"`, but not `"ba"`.

Let's try to define this regex language in Scallop!

``` scl
type Regex = Char(char)           // a single character
           | Star(Regex)          // the star of a regex
           | Union(Regex, Regex)  // a union of two regexes
           | Concat(Regex, Regex) // concatenation of two regexes
```

As can be seen, we have defined 4 variants of this regex language.
With this, our regex `"a*b"` can be expressed as follows:

``` scl
// a*b
const A_STAR_B = Concat(Star(Char('a')), Char('b'))
```

Now, let's define the actual semantics of this regex language and write a relation `matches` to check if the regex matches with a given sub-string.
We first setup the types of such relations.
- `input_regex` is a unary-relation for holding the regex to be checked against;
- `input_string` is a unary-relation for holding the string to be checked against;
- `matches_substr` is for checking if a sub-regex `r` can be matched with the input string between `begin` and `end` indices, where `end` is exclusive;
- `matches` is a boolean relation telling whether the `A_STAR_B` regex matches with the input string or not.

``` scl
type input_regex(r: Regex)
type input_string(s: String)
type matches_substr(r: Regex, begin: usize, end: usize)
type matches()
```

The main bulk of the code will then be dedicated to define the `matches_substr` relation.
At a high level, we decompose on each type of regex, and match on sub-strings.
The first rule that we are going to write would be for the `Char` variant.

``` scl
rel matches_substr(r, i, i + 1) = case r is Char(c) and input_string(s) and string_chars(s, i, c)
```

The rule suggests that if the regex `r` is a single character `c`, then we go into the input string `s` and find all the index `i` such that its corresponding character is `c`.
The matched sub-string would start at index `i` and end at index `i + 1`.
Note that the `string_chars` relation is a foreign predicate that decomposes the string into characters.

Similarly, we can write the rules for other variants:

``` scl
// For star; it matches empty sub-strings [i, i) and recursively on sub-regex
rel matches_substr(r, i, i) = case r is Star(_) and input_string(s) and string_chars(s, i, _)
rel matches_substr(r, b, e) = case r is Star(r1) and matches_substr(r, b, c) and matches_substr(r1, c, e)

// For union; any string that matches left or right sub-regex would match the union
rel matches_substr(r, b, e) = case r is Union(r1, r2) and matches_substr(r1, b, e)
rel matches_substr(r, b, e) = case r is Union(r1, r2) and matches_substr(r2, b, e)

// For concat; we need strings to match in a consecutive matter
rel matches_substr(r, b, e) = case r is Concat(r1, r2) and matches_substr(r1, b, c) and matches_substr(r2, c, e)
```

Lastly, we add the rule to derive the final `matches` relation.
Basically, it checks if the regex matches the start-to-end of the input string

``` scl
rel matches() = input_regex(r) and input_string(s) and matches_substr(r, 0, $string_length(s))
```

Let us test the result!

``` scl
rel input_regex(A_STAR_B)
rel input_string("aaaab")
query matches // {()}
```

## Dynamically Creating Entities

There are cases where we want to create new entities during the deductive process.
This is done through the `new` keyword followed by the entity to create.
Suppose we have the definition of `List` and some pretty-printing code for it:

``` scl
type List = Cons(i32, List) | Nil()

rel to_string_2(l, "]")                      = case l is Nil()
rel to_string_2(l, $format("{}]", i))        = case l is Cons(i, Nil())
rel to_string_2(l, $format("{}, {}", i, ts)) = case l is Cons(i, tl) and case tl is Cons(_, _) and to_string_2(tl, ts)
rel to_string(l, $format("[{}", tl))         = to_string_2(l, tl)
```

The following example shows that, given an input list `l`, we generate a result list `Cons(1, l)`.

``` scl
type input_list(List)
rel result_list(new Cons(1, l)) = input_list(l)
```

Given an actual list defined as a constant, we will be able to specify that the constant is the input list:

``` scl
const MY_INPUT_LIST = Cons(2, Cons(3, Nil()))
rel input_list(MY_INPUT_LIST)
```

Now, let's visualize the results!

``` scl
rel input_list_str(s) = to_string(MY_INPUT_LIST, s)
rel result_list_str(s) = result_list(l) and to_string(l, s)

query input_list_str  // [2, 3]
query result_list_str // [1, 2, 3]
```

As can be seen, through the `new` operator, we have essentially created a new list containing the element `1`.
We note that the rule for `result_list` is *not* recursive.
In general, extra care needs to be taken to ensure that the program does not go into infinite loop.`

### Case Study: Creating Entities for Equality Saturation

In this case study we look at the problem of equality saturation.
Given an symbolic expression, there might be ways to simplify it, which are defined through *rewrite rules*.
Notice that after simplification, the program should be equivalent to the input.
The problem is challenging as there might be multiple ways to apply the rewrite rules.
How do we then systematically derive the simplest equivalent program?

A simple example here is the symbolic arithmetic expression language, with constant, variables, and summation rule:

``` scl
type Expr = Const(i32) | Var(String) | Add(Expr, Expr)
```

One example expression that we can express in this language would be

``` scl
const MY_EXPR = Add(Add(Const(-3), Var("a")), Const(3)) // (-3 + a) + 3
```

For visualization, we write a `to_string` function

``` scl
rel to_string(p, i as String) = case p is Const(i)
rel to_string(p, v)           = case p is Var(v)
rel to_string(p, $format("({} + {})", s1, s2)) =
  case p is Add(p1, p2) and to_string(p1, s1) and to_string(p2, s2)
```

If we query on `to_string` for `MY_EXPR`, we would get

``` scl
query to_string(MY_EXPR, s) // s = "((-3 + a) + 3)"
```

Now let us deal with the actual simplification.
The expression `(-3 + a) + 3` could be simplified to just `a`, as the `-3` and `3` cancels out.
The way to do the simplification is to write two things:

1. rewrite rules in the form of equivalence relation;
2. the weight function giving each expression a weight to tell which expression is *simpler*.

For this, the following set of relations needs to be defined.

``` scl
type input_expr(expr: Expr)
type equivalent(expr_1: Expr, expr_2: Expr)
type weight(expr: Expr, w: i32)
type simplest(expr: Expr)
```

Note that we need set a prior knowledge on `equivalent`: the `expr_1` is always *more complex* than the `expr_2`.
This is to prevent the simplification to go to arbitrary direction and result in infinite-loop.
In such case, `equivalent` would not be commutative.
Let us start with `equivalent` and define its basic property of identity and transitivity:

``` scl
// Identity
rel equivalent(e, e) = case e is Const(_) or case e is Var(_) or case e is Add(_, _)

// Transitivity
rel equivalent(e1, e3) = equivalent(e1, e2) and equivalent(e2, e3)
```

Now, we can write the rewrite rules.
The first one we are going to write states that, if `e1` and `e1p` are equivalent and `e2` and `e2p` are equivalent,
their additions (`Add(e1, e2)` and `Add(e1p, e2p)`) are equivalent too.

``` scl
// e1 == e1p, e2 == e2p ==> (e1 + e2) == (e1p + e2p)
rel equivalent(e, new Add(e1p, e2p)) = case e is Add(e1, e2) and equivalent(e1, e1p) and equivalent(e2, e2p)
```

The next rule states that Addition is commutative, such that `Add(a, b)` is equivalent to `Add(b, a)`:

``` scl
// (a + b) == (b + a)
rel equivalent(e, new Add(b, a)) = case e is Add(a, b)
```

We also have a rule for associativity:

``` scl
// (a + (b + c)) == ((a + b) + c)
rel equivalent(e, new Add(new Add(a, b), c)) = case e is Add(a, Add(b, c))
```

A rule for simplifying adding summation identity 0:

``` scl
// a + 0 = a
rel equivalent(e, a) = case e is Add(a, Const(0))
```

A rule for reducing two constants addition:

``` scl
rel equivalent(e, Const(a + b)) = case e is Add(Const(a), Const(b))
```

Now we have 5 rewrite-rules in place, let us define how to compute the weight of each expression.
The leaf nodes (`Var` and `Const`) have weight of `1`, and the addition have the weight from left and right sub-expr added together plus 1.

``` scl
rel weight(e, 1) = case e is Var(_) or case e is Const(_)
rel weight(e, l + r + 1) = case e is Add(a, b) and weight(a, l) and weight(b, r)
```

Lastly, we use the aggregation to find the equivalent programs with the minimum weight, which is our definition of the "simplest" program.
Note that we have used an `argmax` aggregation denoted by `min[p]` here:

``` scl
rel best_program(p) = p := argmin[p](w: input_expr(e) and equivalent(e, p) and weight(p, w))
```

If we query for the best program and turn it into string, we will get our expected output, a single variable `"a"`!

``` scl
rel best_program_str(s) = best_program(p) and to_string(p, s)
query best_program_str // {("a")}
```

## Parsing Entities from String

Scallop provides foreign functions and predicates for dynamically parsing entities from string input.
Consider the following example:

``` scl
type Expr = Const(f32) | Add(Expr, Expr)

rel expr_str = {"Add(Const(1), Const(2.5))"}
```

Let us say that we want to parse an expression from the `expr_str`, we can do the following:

``` scl
rel expr($parse_entity(s)) = expr_str(s)
```

Here, we are using the foreign function of `$parse_entity`.
We would get the following result:

``` scl
query expr
// expr: {(entity(0xadea13a2621dd155))}
```
