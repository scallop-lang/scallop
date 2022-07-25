# Design of Group By

## Examples

### Example 1

The following count does not have a group by variable

``` scl
rel num_cars(n) :- n = count(o: is_a(o, "car"))
```

### Example 2

The following count does have a group by variable `c`.
The body of the rule `is_a(o, "car"), color(o, c)` bounds two variables: `o` and `c`.
We want the variables that occur in the head that is not "to-aggregate" or "argument" values.

``` scl
rel num_cars_of_color(c, n) :- n = count(o: is_a(o, "car"), color(o, c))
```

### Example 3

The following count does have a group by variable `c`, note that `s` is not a group-by variable:

``` scl
rel num_cars_of_color(c, n) :- n = count(o: is_a(o, "car"), color(o, c), shape(o, s))
```

Although we have `shape(o, s)`, but we are not storing `s` in the head.
Therefore we do not treat `s` as a group-by variable.

### Example 4

``` scl
rel num_cars_of_color(c, n) :- n = count(o: is_a(o, "car"), color(o, c) where c: all_colors(c))
```

body_bounded_vars: o, c
group_by_bounded_vars: c
group_by_vars: c

### Example 5

``` scl
rel eval_yn(e, b) :- b = exists(o: eval_obj(f, o) where e: exists_expr(e, f))
```

body_bounded_vars: f, o
group_by_bounded_vars: e, f
group_by_vars: e
to_agg_vars: o
