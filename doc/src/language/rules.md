# Rules

*Rules* are the fundamental to computation in Scallop.
Each rule defines the value and data flowing from some relation to another relation.
In the following program, we have defined a few facts for the `edge` relation.
On the second line, we have defined that, for each edge `(a, b)`, there is also a path `(a, b)`.
We note that here, `a` and `b` are variables instead of constants as we have with defining facts.
During computation, the two facts in `edge` will populate the `path` relation.
This way, we have defined a rule for the `path`, which is executed during computation.

``` scl
rel edge = {(0, 1), (1, 2)}
rel path(a, b) = edge(a, b) // (0, 1), (1, 2)
```

In this section, we talk about how we write rules in Scallop and how intricate computation can be done through it.

## Syntax

In general, the basic rules in Scallop are of the form

```
RULE    ::= rel ATOM = FORMULA
FORMULA ::= ATOM
          | not ATOM
          | CONSTRAINT
          | AGGREGATION
          | FORMULA and FORMULA
          | FORMULA or FORMULA
          | ( FORMULA )
```

For each rule, we name the atom on the left to be the *head* of the rule, and the formula on the right to be the *body*.
We read it from right to left: when the body formula holds, the head also holds.
The formula might contain atoms, negated atoms, aggregations, conjunction, disjunction, and a few more constructs.
For this section, we focus on simple (positive) atom, constraints, and their conjunctions and disjunctions.
We will leave the discussion of negation and aggregation to the next sections.

## Atom

Simple atoms are of the form `RELATION(ARG_1, ARG_2, ...)`.
Similar to facts, we have the relation name followed by a tuple of numerous arguments.
Now, the arguments can be of richer forms, involving variables, constants, expressions, function calls, and many more.

Considering the most basic example from above:

``` scl
rel path(a, b) = edge(a, b)
```

We have two variables `a` and `b` *grounded* by the `edge` relation.
This means we are treating the variables `a` and `b` as source of information, which can be propagated to the head.
In this example, the head also contains two variables, both being grounded by the body.
Therefore the whole rule is well formed.

In case the head variables are not grounded by the body, such as the following,

``` scl
rel path(a, c) = edge(a, b)
```

we would get an error that looks like the following:

```
[Error] Argument of the head of a rule is ungrounded
  REPL:1 | rel path(a, c) = edge(a, b)
         |             ^
```

The error message points us to the variable `c` that has not being grounded in the body.

For basic atoms, such as the ones that the user has defined, can be used to directly ground variables which are directly arguments of the atoms.
They can be used to ground other variables or expressions.
In the following example, although the rule itself might not make any sense, the variable `a` is used to ground the expression `a + 1`.
Therefore, the rule is completely valid.

``` scl
rel output_relation(a, a + 1) = input_relation(a)
```

In certain cases, expressions can be used to bound variables as well!

``` scl
rel output_relation(a, b) = input_relation(a, b + 1)
```

In the above example, the expression `b + 1` can be used to derive `b`, and thus making the variable `b` grounded.
However, this might not be true for other expressions:

``` scl
rel output_relation(b, c) = input_relation(b + c) // FAILURE
```

The `input_relation` can ground the expression `b + c` directly, however, the two arguments `b` and `c` cannot be derived from their sum, as there are (theoretically) infinite amount of combinations.
In this case, we will get a compilation failure.

There can be constraints present in atoms as well.
For example, consider the following rule:

``` scl
rel self_edge(a) = edge(a, a)
```

The atom `edge(a, a)` in the body grounds only one variable `a`.
But the pattern is used to match any edge that goes from `a` and to `a` itself.
Therefore, instead of grounding two values representing the "from" and "to" of an `edge`, we are additionally posing constraint on the type of edge that we are matching.
Conceptually, we can view the above rule as the following equivalent rule:

``` scl
rel self_edge(a) = edge(a, b) and a == b
```

where there is an additional constraint posed on the equality of `a` and `b`.
We are going to touch on `and` and constraints in the upcoming sections.

## Disjunction (Or)

The body formula can contain logical connectives such as `and`, `or`, `not`, and `implies`, used to connect basic formulas such as *Atom*.
In the following example, we are defining that if `a` is `b`'s *father* or *mother*, then `a` is `b`'s parent:

``` scl
rel parent(a, b) = father(a, b)
rel parent(a, b) = mother(a, b)
```

In this program, we have divided the derivation of `parent` into two separate rules, one processing the `father` relationship and the other processing the `mother` relationship.
This natually form a disjunction (or), as the derivation of `parent` can come from 2 disjunctive sources.
Note that in Scallop (or Datalog in general), the ordering of the two rules does not matter.

Therefore, given that

``` scl
rel father = {("Bob", "Alice")}
rel mother = {("Christine", "Alice")}
```

we can derive that the `parent` relation holding two tuples, `("Bob", "Alice")` and `("Christine", "Alice")`.

The above program can be rewritten into a more compact form that looks like the following:

``` scl
rel parent(a, b) = father(a, b) or mother(a, b)
// or
rel parent(a, b) = father(a, b) \/ mother(a, b)
```

We have used an explicit `or` (`\/`) keyword to connect the two atoms, `father(a, b)` and `mother(a, b)`.
The `\/` symbol, which is commonly seen in the formal logics as the symbol vee (\\(\vee\\)), is also supported.
Notice that written in this way, each branch of the disjunction need to fully bound the variables/expressions in the head.

## Conjunction (And)

To demonstrate the use of `and`, let's look at the following example computing the relation of `grandmother` based on `father` and `mother`:

``` scl
rel grandmother(a, c) = mother(a, b) and father(b, c)
// or
rel grandmother(a, c) = mother(a, b) /\ father(b, c)
```

Notice that the symbol `/\` is a replacement for the `and` operator, which resembles the wedge (\\(\wedge\\)) symbol seen in formal logics.

As can be seen from the rule, the body grounds three variables `a`, `b`, and `c`.
The variables `a` and `b` comes from `mother` and the variables `b` and `c` comes from `father`.
Notice that there is one variable, `b`, in common.
In this case, we are *joining* the relation of `mother` and `father` on the variable `b`.

## Constraints

Rule body can have boolean constraints.
For example, the conjunctive rule above can be re-written as

``` scl
rel grandmother(a, c) = mother(a, b) and father(bp, c) and b == bp
```

Here, we are posing an equality (`==`) constraint on `b` and `bp`.
Normally, constraints are such kind of binary expressions involving predicates such as

- equality and inequality (`==` and `!=`)
- numerical comparisons (`<`, `>`, `<=`, and `>=`)

## Other constructs

There are other constructs available for defining rules, which we continue to discuss in detail in other sections:

- [Disjunctive head](disj_conj_head.md)
- [Recursive Rules](recursion.md)
- [Negation](negation.md)
- [Aggregation](aggregation.md)
- [Foreign Predicates](foreign_predicates.md)

## Traditional Datalog Syntax

If you are familiar with traditional Datalog, you can have it by swapping the `=` with `:-`, and the `and` to `,`
For example, the rule for defining `grandmother` can be rewritten as

``` scl
rel grandmother(a, c) :- mother(a, b), father(b, c)
```
