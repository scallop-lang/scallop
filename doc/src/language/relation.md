# Relations and Facts

Scallop is a relational and logical programming language.
As described in the [Wikipedia](https://en.wikipedia.org/wiki/Logic_programming):

> Logic programming is a programming paradigm which is largely based on formal logic.
> Any program written in a logic programming language is a set of sentences in logical form,
> expressing facts and rules about some problem domain.

In Scallop, relations are the most fundamental building blocks of program.
In the following example, we have declared the type of a relation called `edge`, using the `type` keyword:

``` scl
type edge(a: i32, b: i32)
```

We say that the name `edge` is a *predicate* or a *relation*.
Inside of the parenthesis, we have two `arguments`, `a: i32` and `b: i32`.
Therefore, we have `edge` being an *arity-2* relation, due to it having 2 arguments.
For the argument `a: i32`, we give a name of the field (`a`) and a type of that argument `i32`.
Here, both of the arguments are of the `i32` type, which means signed-*i*nteger, *32*-bit.
For more information on value types, refer to [the Value Types section](#value-types).

The above line only declares the type of the relation but not the *content* of the relation.
The actual information stored in the relations are called *facts*.
Here we define a single fact under the relation `edge`:

``` scl
rel edge(0, 1)
```

Assuming `0` and `1` each denote an ID of a node, this fact declares that there is an edge going from node `0` to node `1`.
There are two arguments in this fact, matching the arity of this relation.
Regardless of the predicate `edge`, one also simply consider the `(0, 1)` as a *tuple*, more specifically, a *2-tuple*.

To declare multiple facts, one can simply write multiple single fact declaration using the `rel` keyword, like

``` scl
rel edge(0, 1)
rel edge(1, 2)
```

One can also use the *set* syntax to declare multiple facts of a relation.
The following line reads: "the relation `edge` contains a set of tuples, including `(0, 1)` and `(1, 2)`":

``` scl
rel edge = {(0, 1), (1, 2)}
```

Note that it is possible to declare multiple fact sets for the same relation.

``` scl
rel edge = {(0, 1), (1, 2)}
rel edge = {(2, 3)}
```

With the above two lines the edge relation now contains 3 facts, `(0, 1)`, `(1, 2)`, and `(2, 3)`.

## Examples of Relations

### Boolean and 0-arity Relation

Many things can be represented as relations.
We start with the most basic programming construct, boolean.
While Scallop allows value to have the boolean type, relations themselves can encode boolean values.
The following example contains an *arity-0* relation named `is_target`:

``` scl
type is_target()
```

There is only one possible tuple that could form a fact in this relation, that is the *empty tuple* `()`.
Assuming that we are treating the relation `is_target` as a set, then if the set contains no element (i.e., empty), then it encodes boolean "false".
Otherwise, if the set contains exactly one (note: it can contain at most one) tuple, then it encodes boolean "true".
Declaring only the type of `is_target` as above, would assume that the relation is empty.
To declare the fact, we can do:

``` scl
rel is_target()
// or
rel is_target = {()}
```

### Unary Relations

Unary relations are relations of arity 1.
We can define unary relations for "variables" as we see in other programming languages.
The following example declares a relation named `greeting` containing one single string of `"hello world!"`.

``` scl
rel greeting("hello world!")
// or
rel greeting = {("hello world!",)}
```

Note that for the second way of expressing the fact, we may omit the parenthesis and make it cleaner:

``` scl
rel greeting = {"hello world!"}
```

In light of this, we may write the following rule:

``` scl
rel possible_digit = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
```

### Integer Arithmetics as Relations

Integer arithmetics can be represented as relations as well.
Consider a simple summation in algebra, `a + b = c` encodes the sum relationship among two operands (`a` and `b`) and their summation (`c`).
Encoded in Scallop, they form arity-3 relations:

``` scl
type add(op1: i32, op2: i32, result: i32)
```

Note that, in Scallop, relations are *not* polymorphic.
That is, every relation, no matter declared or inferred, only has one type annotation.

> We are working on an update in the future to relax this restriction.

To declare facts of this `add` relation, such as `3 + 4 = 7`, we write

``` scl
rel add(3, 4, 7) // 3 + 4 = 7
```

However, you might notice that the `add` relation is theoretically *infinite*.
That is, there are infinite amount of facts that can satisfy the `add` relation.
There is no way that we can possibly enumerate or declare all the facts.
In such case, we resort to declaring rules using foreign functions or predicates, which we will discuss later.
For now, let's use `add` as an example relation that encodes integer arithmetics.

### Terminologies

We have the following terminologies for describing relations.

- Boolean Relation: arity-0 relation
- Unary Relation: arity-1 relation
- Binary Relation: arity-2 relation
- Ternary Relation: arity-3 relation

## Type Inference

Scallop supports *Type Inference*.
One does not need to fully annotate every relation on their types.
Types are inferred during the compilation process.

For example, given the following code,

``` scl
rel edge = {(0, 1), (1, 2)}
```

we can infer that the relation `edge` is a binary-relation where both arguments are integers.
Note that when integers are specified, they are set default to the type of `i32`.

Type inference will fail if conflicts are detected.
In the following snippet, we have the second argument being `1` as integer and also `"1"` as string.

``` scl
rel edge = {(0, 1), (0, "1")}
```

Having this code will raise the following compile error, suggesting that the types cannot be unified.
Note that the following response is generated in `sclrepl` command line interface.

```
[Error] cannot unify types `numeric` and `string`, where the first is declared here
  REPL:0 | rel edge = {(0, 1), (0, "1")}
         |                 ^
and the second is declared here
  REPL:0 | rel edge = {(0, 1), (0, "1")}
         |                         ^^^
```

For more information on values and types, please refer to the [next section](value_type.md)
