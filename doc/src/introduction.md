# Scallop, a Language for Neurosymbolic Programming

<center>
  <img src="res/img/scallop-logo-ws-512.png" width="200px" />
</center>

Scallop is a language based on DataLog that supports differentiable logical and relational reasoning.
Scallop program can be easily integrated in Python and even with a PyTorch learning module.
You can also use it as another DataLog solver.
This book aims to give both high-level overview of the language usage and also low-level documentation on how each language feature is used.

The following example shows how knowledge base facts, rules, and probabilistic facts recognized from images can operate together.

``` scl
// Knowledge base facts
rel is_a("giraffe", "mammal")
rel is_a("tiger", "mammal")
rel is_a("mammal", "animal")

// Knowledge base rules
rel name(a, b) :- name(a, c), is_a(c, b)

// Recognized from an image, maybe probabilistic
rel name = {
  0.3::(1, "giraffe"),
  0.7::(1, "tiger"),
  0.9::(2, "giraffe"),
  0.1::(2, "tiger"),
}

// Count the animals
rel num_animals(n) :- n = count(o: name(o, "animal"))
```

## Table of Content

Please refer to the side-bar for a detailed table of content.
At a high-level, we organize this book into the following 5 sections:

### Installation and Crash Course

[Installation](installation.md) gives instructions on how to install the Scallop on your machine.
[Crash Course](crash_course.md) gives a quick introduction to what the language is and how it is used.
Both sections are designed so that you can start quickly with Scallop.

### Scallop and Logic Programming

[Scallop and Logic Programming](language/index.md) aims to give you a detailed introduction on the language.
It introduces language features such as relational programming, negation and aggregation, queries, foreign constructs, and etc.
Reading through all of these you should be well-versed in Scallop's core functionality and you will be able to use Scallop as a Datalog engine.

``` scl
type fib(bound x: i32, y: i32)
rel fib = {(0, 1), (1, 1)}
rel fib(x, y1 + y2) = fib(x - 1, y1) and fib(x - 2, y2) and x > 1
query fib(10, y)
```

### Scallop and Probabilistic Programming

[Scallop and Probabilistic Programming](probabilistic/index.md) introduces the probabilistic side of Scallop.
You will learn to tag facts with probabilities, its underlying algorithms and frameworks, and additional programming constructs for probabilistic semantics.
By the end of this section, you will be familiar with using Scallop as a probabilistic programming language.

``` scl
rel attr = { 0.99::(OBJECT_A, "blue"), 0.01::(OBJECT_B, "red"), ... }
rel relate = { 0.01::(OBJECT_A, "holds", OBJECT_B), ... }
```

### Scallopy and Neurosymbolic Programming

[Scallopy and Neurosymbolic Programming](scallopy/index.md) goes into the heart of Scallop to introduce applying Scallop to write Neurosymbolic applications.
Neurosymbolic methods are for methods that have both neural and logical components.
For this, we are going to use the Python binding of Scallop, `scallopy`, to integrate with machine learning libraries such as PyTorch.
This section will be describing the API of `scallopy`.

``` py
sum_2 = scallopy.Module(
  program="""type digit_1(i32), digit_2(i32)
             rel sum_2(a + b) = digit_1(a) and digit_2(b)""",
  input_mappings={"digit_1": range(10), "digit_2": range(10)},
  output_mapping=("sum_2", range(19)))
```

### For Developers

[For Developers](developer/index.md) discusses how developers and researchers who are interested in extending Scallop can step into the source code of Scallop and program extensions.
