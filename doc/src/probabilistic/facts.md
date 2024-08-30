# Fact with Probability

We can associate facts with probabilities through the use of `::` symbol.
This can be done with the set syntax and also individual fact syntax:

``` scl
rel color = {0.1::"red", 0.8::"green", 0.1::"blue"}

// or

rel 0.1::color("red")
rel 0.8::color("green")
rel 0.1::color("blue")
```

## Mutual exclusive facts

Within the set annotation, if we replace the comma (`,`) with semi-colons (`;`), we will be specifying mutual exclusive facts.
If one is encoding a categorical distribution, they should be specifying mutual exclusions by default.
Suppose we have two MNIST digits that can be classified as a number between 0 to 9.
If we represent each digit with their ID, say `A` and `B`, we should write the following program in Scallop:

``` scl
type ImageID = A | B
type digit(img_id: ImageID, number: i32)

rel digit = {0.01::(A, 0); 0.86::(A, 1); ...; 0.03::(A, 9)}
rel digit = {0.75::(B, 0); 0.03::(B, 1); ...; 0.02::(B, 9)}
```

Notice that we have specified two sets of digits, each being a mutual exclusion, as suggested by the semi-colon separator (`;`).
This means that each of `A` and `B` could be classified as one of the 10 numbers, but not multiple.

## Specifyin mutually exclusive facts in Scallopy
