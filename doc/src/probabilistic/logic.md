# Probabilistic Rules

In Scallop, rules can have probabilities too, just like the probabilities associated with the facts and tuples.
For instance, you might write the following probabilistic rule to denote that "when earthquake happens, there is a 80% chance that an alarm will go off":

``` scl
rel 0.8::alarm() = earthquake()
```

Combine the above rule with the fact that earthquake happens with a 10% probability, we obtain that the alarm will go off with a 0.08 probability.
Note that this result can be obtained using `topkproofs` or `addmultprob` provenance, while a provenance such as `minmaxprob` will give different results.

``` scl
rel 0.1::earthquake()

query alarm // 0.08::alarm()
```

## Rule tags with expressions

What is special about the probabilities of rules is that the probabilities could be expressions depending on values within the rule.
For instance, here is a set of rules that say that the probability of a path depends on the length of a path, which falls off when the length increases:

``` scl
// A few edges
rel edge = {(0, 1), (1, 2)}

// Compute the length of the paths (note that we encode length with floating point numbers)
rel path(x, y, 1.0) = edge(x, y)
rel path(x, z, l + 1.0) = path(x, y, l) and edge(y, z)

// Compute the probabilistic path using the fall off (1 / length)
rel 1.0 / l :: prob_path(x, y) = path(x, y, l)

// Perform the query with arbitrary
query prob_path // prob_path: {1.0::(0, 1), 0.5::(0, 2), 1.0::(1, 2)}
```

Here, since `path(0, 1)` and `path(1, 2)` have length 1, their probability is `1 / 1 = 1`.
However, `path(0, 2)` has length 2 so its probability is `1 / 2 = 0.5`.

As can be seen, with the support for having expressions in the tag, we can encode more custom probabilistic rules in Scallop.
Internally, this is implemented through the use of custom foreign predicates.

## Rule tags that are not floating points

In general, Scallop supports many forms of tag, including but not limited to probabilities (floating points).
For instance, we can encode boolean as well:

``` scl
rel constraint(x == y) = digit_1(x) and digit_2(y)
rel b::sat() = constraint(b)
```

The relation `constraint` has type `(bool)`, and therefore the variable `b` in the second rule has type boolean as well.
With the second rule, we lift the boolean value into the boolean tag associated with the nullary relation `sat`.

## Associating rules with tags from Scallopy

> We elaborate on this topic in the Scallopy section as well

You can associate rules with tags from Scallopy as well, so that we are not confined to Scallop's syntax.
For instance, the following python program creates a new Scallop context and inserts a rule with a tag of 0.8.

``` py
ctx = scallopy.Context(provenance="topkproofs")
ctx.add_rule("alarm() = earthquake()", tag=0.8)
```

Of course, the tag doesn't need to be a simple constant floating point, since we are operating within the domain of Python.
How about using a PyTorch tensor? Certainly!

``` py
ctx = scallopy.Context(provenance="topkproofs")
ctx.add_rule("alarm() = earthquake()", tag=torch.tensor(0.8, requires_grad=True))
```

Notice that we have specified that `requires_grad=True`.
This means that if any Scallop output depends on the tag of this rule, the PyTorch back-propagation will be able to accumulate gradient on this tensor of 0.8.
Any optimization will have an effect on updating the tag, by essentially treating it as a parameter.
Of course, we might need more thoughts so that the optimization can actually happen.
For instance, you will need to tell the optimizer that this tensor is a parameter.
But we will delay this discussion to a later section.
