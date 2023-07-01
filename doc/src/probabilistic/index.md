# Scallop and Probabilistic Programming

One fundamental concept in machine learning is *probability*.
Scallop, being a neurosymbolic programming language, supports probability and probabilistic programming natively.
For example, one can write the following program:

``` scl
type Action = UP | DOWN | LEFT | RIGHT
rel predicted_action = {0.05::UP, 0.09::DOWN, 0.82::LEFT, 0.04::RIGHT}
```

where the `predicted_action` relation encodes a distribution of actions and their probabilities.
In particular, the `UP` action is predicted to have a \\(0.05\\) probability.
Here, the `::` symbol is used to suggest that probabilities (such as 0.05) are used to *tag* the facts (such as `UP`).

Since we can define probability on user declared facts, the derivation of new facts will be associated with probabilities too.
This means that Scallop is doing *probabilistic reasoning*.
The whole probabilistic reasoning semantics of Scallop is defined with the theory of *provenance semiring*.
In this chapter, we give detailed explanation to the probabilities appeared in Scallop.
