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
Notice that `range` can be applied on any integer data, making it a generic predicate.
For example, to use `range` on `i32` data, we will need to invoke `range<i32>`:

``` scl
rel result(x) = range<i32>(0, 5, x)
```

Here we enumerate the value of `x` from 0 (inclusive) to 5 (exclusive), meaning that we will obtain that `result = {0, 1, 2, 3, 4}`.
For the rest of this section, we describe in detail how foreign predicates are constructed in Scallop and why are they useful.

## Foreign Predicate Types

Foreign predicates can be generic and are statically typed.
In addition to just providing the argument types, we also need to provide a boundness pattern.

A boundness pattern is a string of length equal to the relation arity and consisting of `b` and `f`.
The character `b` means *bounded*, reflecting the variable on that position is taken as input to the predicate.
The character `f` means *free*, suggesting that the variable on that position will be generated as output by the predicate.

For example, the full definition of `range` is

```
range<T: Integer>(begin: T, end: T, i: T)[bbf]
```

Notice that at the end of the definition we have `[bbf]`.
Here, `bbf` is a boundness pattern for range, suggesting that `begin` and `end` will be provided as input, and `i` will be generated as output.

> In the future, we plan to allow the definition of multiple boundness patterns

## Standard Library of Foreign Predicates (Part A)

In this part, we give an overview to the foreign predicates that are discrete only.

| Foreign Predicate | Description |
|-------------------|-------------|
| `datetime_ymd(d: DateTime, y: i32, m: u32, d: u32)[bfff]` | Get the `y`ear, `m`onth, and `day` from a `DateTime` value |
| `range<T: Integer>(begin: T, end: T, i: T)[bbf]` | Generate all the integers `i` starting from `begin` and end with `end - 1` |
| `string_chars(s: String, i: usize, c: char)[bff]` | Generate all the index-character tuples inside of string `s` |
| `string_find(s: String, pat: String, begin: usize, end: usize)[bbff]` | Generate all the begin-end ranges of the pattern `pat`'s occurrence in the string `s` |
| `string_split(s: String, pat: String, out: String)[bbf]` | Split the string `s` using the pattern `pat` and generate the `out` strings |
