# Aggregation with Probability

With the introduction of probabilities, many existing aggregators are augmented with new semantics, which we typically call *multi-world semantics*.
What's more, there are new aggregators, such as `softmax`, `rank`, and `weighted_avg`, that make use of the probabilities.
We introduce these aggregators one-by-one in this section.

## Multi-world Semantics with Aggregators

Let us take the `count` aggregator as an example.
Consider we have 2 objects, each could be big or small with their respective probabilities:

``` scl
type OBJ = OBJ_A | OBJ_B
rel size = {0.8::(OBJ_A, "big"); 0.2::(OBJ_A, "small")} // obj A is very likely big
rel size = {0.1::(OBJ_B, "big"); 0.9::(OBJ_B, "small")} // obj B is very likely small
```

Now let's say we want to count how many big objects are there, by using the following

Note that even when using probabilites, one can opt to not use the multi-world semantics by adding `!` sign to the end of the aggregator.

## New Aggregators using Probabilities

### Softmax and Normalize

### Rank

### Weighted Average and Weighted Sum
