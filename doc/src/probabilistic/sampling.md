# Sampling with Probability

In Scallop, samplers share the same syntax as aggregators.
They usually work with probabilistic provenances, but can also work without them.
Here are some example samplers:

- `top`: get the $k$ facts with top probabilities
- `categorical`: treat the relation as a categorical distribution and sample from it
- `uniform`: treat the relation as a uniform distribution and sample from it

Let's take `top` as an example.
We can obtain the top ranked symbol by using the following rule:

``` scl
rel symbols = {0.9::"+", 0.05::"-", 0.02::"3"}
rel top_symbol(s) = s := top<1>(s: symbols(s)) // 0.9::top_symbol("+")
```
