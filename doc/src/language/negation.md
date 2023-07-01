# Negations

Scallop supports negation to be attached to atoms to form negations.
In the following example, we are trying to obtain the set of people with no children:

``` scl
rel person = {"bob", "alice", "christine"} // There are three persons of interest
rel father = {("bob", "alice")}            // Bob is Alice's father
rel mother = {("alice", "christine")}      // Alice is Christine's mother

rel has_no_child(n) = person(n) and not father(n, _) and not mother(n, _)
```

The last rule basically says that if there is a person `n` who is neither anyone's father nor anyone's mother then the person `n` has no child.
This is indeed what we are going to obtain:

```
has_no_child: {("christine",)}
```

It is clear that negations are very helpful in writing such kind of the rules.
However, there are many restrictions on negations.
We explain in detail such restrictions.

## Negation and Variable Grounding

If we look closely to the rule of `has_no_child` above, we will find that there is an atom `person(n)` being used in the body.
Why can't we remove it and just say "if one is neither father nor mother then the one has no child"?

``` scl
rel has_no_child(n) = not father(n, _) and not mother(n, _) // Error: variable `n` is not grounded
```

The problem is with variable grounding.
For the variable `n` to be appeared in the head, there is **no positive atom** that grounds it.
All we are saying are what `n` is not, but not what `n` is.
With only "what it is not", it could be literally anything else in the world.

Therefore, we need to ground it with a positive atom such as `person(n)`.
With this rule, we have basically

## Stratified Negation

Expanding upon [our definition of dependency graph](recursion.md#relation-dependency),
if a predicate occurs in a negative atom in a body,
we say that the predicate of the rule head *negatively depends* on this predicate.
For example, the above `has_no_child` example has the following dependency graph.
Notice that we have marked the *positive* (`pos`) and *negative* (`neg`) on each edge:

```
person <--pos-- has_no_child --neg--> father
                      |
                      +-----neg-----> mother
```

Scallop supports *stratified negation*, which states that there is never a loop in the dependency graph which involves a negative dependency edge.
In other words, if there exists such a loop, the program will be rejected by the Scallop compiler.
Consider the following example:

``` scl
rel is_true() = not is_true() // Rejected
```

The relation `is_true` negatively depends on the relation `is_true` itself, making it a loop containing a negative dependency edge.
The error message would show that this program "cannot be stratified".
If we draw the dependency graph of this program, it look like the following:

``` scl
is_true <---+
   |        |
   +--neg---+
```

Since there is a loop (`is_true -> is_true`) and the loop contains a negative edge, this program cannot be stratified.

The reason that stratified negation is named such way is that, if there is no negative dependency edge in a loop, the whole dependency can be decomposed in to [*strongly connected components*](https://en.wikipedia.org/wiki/Strongly_connected_component), where inside of each strongly connected component (SCC), there is no negative dependency.
In other words, the negation has been *stratified*, so that the negative edge can only happen between SCCs.
We call each SCC a *stratum*, and the collection of them a *strata*.
Any non-recursive program has a dependency graph forming a *Directed Acyclic Graph* (DAG), and is therefore always stratifiable.

The following program, although containing both negation and recursion, can be stratified:

``` scl
rel path(a, b) = edge(a, b) and not sanitized(b)
rel path(a, c) = path(a, b) and edge(b, c) and not sanitized(b)
```

For it, the following dependency graph can be drawn:

```
sanitized <--neg-- path <----+
                   |  |      |
     edge <--pos---+  +--pos-+
```

In this program, we have three SCCs (or strata):

- Stratum 1: `{edge}`
- Stratum 2: `{sanitized}`
- Stratum 3: `{path}`

Negative dependency only occurs between stratum 2 and 3.
Therefore, the program can be accepted.
