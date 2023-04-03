# Foreign Predicates

Foreign predicates aim to provide programmers with extra capabilities with relational predicates.
Traditional Datalog program defines relational predicate using only horn-rules.
Given the assumption that the input database is finite, these derived relational predicates will also be finite.
However, there are many relational predicates that are infinite and could not be easily expressed by horn-rules.
One such example is the `range` relation.
Suppose it is defined as `range(begin, end, i)` where `i` could be between `begin` (inclusive) and `end` (exclusive).
There could be infinitely many triplets, and we cannot simply enumerate all of them.
But if the first two arguments `begin` and `end` are given, we can reasonably enumerate the `i`.

In Scallop, `range` is available to be used as a **foreign predicate**.
Note that Scallop's foreign predicates are currently statically typed and does not support signature overload.
For example, to use `range` on `i32` data, we will need to invoke `range_i32`:

``` scl
rel result(x) = range_i32(0, 5, x)
```

Here we enumerate the value of `x` from 0 (inclusive) to 5 (exclusive), meaning that we will obtain that `result = {0, 1, 2, 3, 4}`.
For the rest of this section, we describe in detail how foreign predicates are constructed in Scallop and why are they useful.

## Bound and Free Pattern

## Foreign Predicates are Statically Typed

## Available Foreign Predicates in `std`
