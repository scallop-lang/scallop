# Scallop

<p align="center">
  <img width="240" height="240" src="docs/icons/scallop-logo-ws-512.png" />
</p>

Scallop is a language based on DataLog that supports differentiable logical and relational reasoning.
Scallop program can be easily integrated in Python and even with a PyTorch learning module. You can also use it as another DataLog solver.
Internally, Scallop is built on a generalized [Provenance Semiring](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1022&context=db_research) framework.
It allows arbitrary semirings to be configured, supporting Scallop to perform discrete logical reasoning, probabilistic reasoning, and differentiable reasoning.

## Example

Here is a simple probabilistic DataLog program that is written in Scallop:

```
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

## How to use

### Prerequisite

Install `rust` with `nightly` channel set to default.

``` bash
$ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
$ rustup default nightly
```

### Download and Build

``` bash
$ git clone https://github.com/Liby99/scallop-v2.git
$ cd scallop-v2
```

The following three binaries are available. Scroll down for more ways
to use Scallop!

``` bash
$ cargo build --release --bin scli # Scallop Interpreter
$ cargo build --release --bin sclc # Scallop Compiler
$ cargo build --release --bin sclrepl # Scallop REPL
```

### Using Scallop Interpreter

Scallop interpreter (`scli`) interprets a scallop program (a file with extension `.scl`).
You can install `scli` to your system using

``` bash
$ cargo install --path etc/scli
```

Then since `scli` is in your system path, you can simply run

``` bash
$ scli examples/animal.scl
```

Note that by default we don't accept probabilistic input.
If your program is proabalistic and you want to obtain the resulting probabilities, do

``` bash
$ scli examples/digit_sum_prob.scl -p minmaxprob
```

Note that the `-p` argument allows you to specify a provenance semiring.
The `minmaxprob` is a simple provenance semiring that allows for probabilistic reasoning.

### Using Scallop REPL

Scallop REPL (`sclrepl`) is an interactive command line interface for you to try various ideas with Scallop.
You can install `sclrepl` to your system using

``` bash
$ cargo install --path etc/sclrepl
```

Then you can run `sclrepl`. You can type scallop commands like the following

``` bash
$ sclrepl
scl> rel edge = {(0, 1), (1, 2)}
scl> rel path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)
scl> query path
path: {(0, 1), (0, 2), (1, 2)}
scl>
```

### Using `scallopy`

`scallopy` is the python binding for Scallop.
It provides an easy to use program construction/execution pipeline.
With `scallopy`, you can write code like this:

``` python
import scallopy

# Create new context (with unit provenance)
ctx = scallopy.ScallopContext()

# Construct the program
ctx.add_relation("edge", (int, int))
ctx.add_facts("edge", [(0, 1), (1, 2)])
ctx.add_rule("path(a, c) = edge(a, c)")
ctx.add_rule("path(a, c) = edge(a, b), path(b, c)")

# Run the program
ctx.run()

# Inspect the result
print(list(ctx.relation("path"))) # [(0, 1), (0, 2), (1, 2)]
```

In addition, `scallopy` can be seamlessly integrated with PyTorch.
Here's how one can write the `mnist_sum_2` task with Scallop:

``` python
class MNISTSum2Net(nn.Module):
  def __init__(self, provenance="difftopkproofs", k):
    super(MNISTSum2Net, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()

    # Scallop Context
    self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    self.scl_ctx.add_relation("digit_1", int, input_mapping=list(range(10)))
    self.scl_ctx.add_relation("digit_2", int, input_mapping=list(range(10)))
    self.scl_ctx.add_rule("sum_2(a + b) = digit_1(a), digit_2(b)")

    # The `sum_2` logical reasoning module
    self.sum_2 = self.scl_ctx.forward_function("sum_2", list(range(19)))

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
    (a_imgs, b_imgs) = x

    # First recognize the two digits
    a_distrs = self.mnist_net(a_imgs)
    b_distrs = self.mnist_net(b_imgs)

    # Then execute the reasoning module; the result is a size 19 tensor
    return self.sum_2(digit_1=a_distrs, digit_2=b_distrs)
```

To install, please do the following (also specified [here](etc/scallopy/readme.md)):

Assume you are inside of the root `scallop` directory.
First, we need to create a virtual environment for Scallop to operate in.

``` bash
# Mac/Linux (venv, requirement: Python 3.8)
$ python3 -m venv .env
$ source .env/bin/activate # if you are using fish, use .env/bin/activate.fish

# Linux (Conda)
$ conda create --name scallop-lab python=3.8 # change the name to whatever you want
$ conda activate scallop-lab
```

And let's install the core dependencies

``` bash
$ pip install maturin
```

With this, we can build our `scallopy` library

``` bash
$ make install-scallopy
```

If succeed, please run some examples just to confirm that `scallopy` is indeed installed successfully.
When doing so (and all of the above), please make sure that you are inside of the virtual environment or
conda environment.

``` bash
$ python etc/scallopy/examples/edge_path.py
```

## Scallop Language

### Fact Declaration

You can declare a single fact using the following syntax.
In each line you define a single atom with every argument being constant.

```
rel digit(0, 1) // non-probabilitic
rel 0.3::digit(0, 1) // probabilistic
```

Alternatively, you can declare a set of facts using the following syntax.

```
rel digit = {
  0.4::(0, 1),
  0.3::(0, 2),
  0.1::(0, 3),
}
```

### Rule Declaration

You can declare rule using traditional datalog syntax:

```
rel path(a, b) :- edge(a, b)
rel path(a, c) :- path(a, b), edge(b, c)
```

Alternatively, you can use a syntax similar to logic programming:

```
rel path(a, c) = edge(a, c) \/ (path(a, b) /\ edge(b, c))
```

Note that `/\` represents conjunction and `\/` represents disjunction.

### Probabilistic Rule

It is possible to declare a probabilistic rule

```
rel 0.3::path(a, b) = edge(a, b)
rel 0.5::path(b, c) = edge(c, b)
```

### Negation

Scallop supports stratified negation, with which you can write a rule like this:

```
scl> rel numbers(x) = x == 0 \/ (numbers(x - 1) /\ x <= 10)
scl> rel odd(1) = numbers(1)
scl> rel odd(x) = odd(x - 2), numbers(x)
scl> rel even(y) = numbers(y), ~odd(y)
scl> query even
even: {(0), (2), (4), (6), (8), (10)}
```

### Aggregation

We support the following aggregations `count`, `min`, `max`, `sum`, and `prod`.
For example, if you want to count the number of animals, you can write

```
scl> rel num_animals(n) :- n = count(o: name(o, "animal"))
scl> query num_animals
num_animals: {(2)}
```

Here `n` is the final count; `o` is the "key" variable that you want to count on;
`name(o, "animal")` is the sub-formula that can pose constraint on `o`.

Naturally, the arguments that are not key and appears in both the sub-formula and
outside of sub-formula will become a `group-by` variable.
The following example counts the number of objects (`n`) of each color (`c`):

```
scl> rel object_color = {(0, "blue"), (1, "green"), (2, "blue")}
scl> rel color_count(c, n) :- n = count(o: object_color(o, c))
scl> query color_count
color_count: {("blue", 2), ("green", 1)}
```

The results says there are two `"blue"` objects and one `"green"` object, as expected.

For the aggregation such as `min` and `max`, it is possible to get the `argmax` and
`argmin` at the same time.
Building up from the previous object-color example, the following rule can extract the
color that has the most number of objects:

```
scl> rel max_color(c) :- _ = max[c](n: color_count(c, n))
scl> query max_color
max_color: {("blue")}
```

Note that we have `max[c]` denoting that we want to get `c` as the argument for `max`.
Also, we use a wildcard `_` on the left hand side of the aggregation denoting that we
don't care about the aggregation result.
The final answer here is `"blue"` since there are 2 of them, which is greater than that
of color `"green"`.

Combining all of these, you can have a query containing group by and argument simultaneously.
The following example builds on a table containing student, their class, and their grade:

```
rel class_student_grade = {
  (0, "tom", 50),
  (0, "jerry", 70),
  (0, "alice", 60),
  (1, "bob", 80),
  (1, "sherry", 90),
  (1, "frank", 30),
}

rel class_top_student(c, s) = _ = max[s](g: class_student_grade(c, s, g))
```

At the end, we will get `{(0, "jerry"), (1, "sherry")}`.
Note that `"jerry"` is the one who got the highest score in class `0` and
`"sherry"` is the one who got the highest score in class `1`.

### Types

Scallop is a statically typed language which employs type inference, which is why
you don't see the type definitions above.
If you want, it is possible to define the type of the relations and even create new type
aliases.
For example,

```
type edge(usize, usize)
```

Defines that the relation `edge` will be a 2-relation and both of the arguments are of
type `usize`, which follows rust's type idiomatic and represents an unsigned 64 bit numbers
(in a 64-bit system).

Scallop supports the following primitive types:

- Signed Integers: `i8`, `i16`, `i32`, `i64`, `i128`, `isize`
- Unsigned Integers: `u8`, `u16`, `u32`, `u64`, `u128`, `usize`
- Floating Points: `f32`, `f64`
- Boolean: `bool`
- Character: `char`
- String:
  - `&str` (static string which could only be used in static Scallop compiler);
  - `String`
  - `Rc<String>` (Reference counted strings, which is the most efficient)

Some example type definition includes

```
type edge(usize, usize)
type obj_color(usize, String) // object is represented by a number, usize, and color is represented as string
type empty() // 0-arity relation
type binary_expr(usize, String, usize, usize) // expr id, operator, lhs expr id, rhs expr id
```

### Subtype and Type Alias

The following snippet shows how you can define subtype.

```
type Symbol <: usize
type ObjectId <: usize
```

### Input/Output

It is possible to define a relation is an input relation that can be loaded from files.

```
@file("example/input_csv/edge.csv")
input edge(usize, usize)
```

Note that in this case it is essential to define the type of the relation.
When loading `.csv` files, we accept extra loading options:

- deliminator: `@file("FILE.csv", deliminator = "\t")` with deliminator set to a tab (`'\t'`)
- has header: `@file("FILE.csv", has_header = true)`. It is default to `false`
- has probability: `@file("FILE.csv", has_probability = true)`. When set to `true`, the first
  column of the CSV file will be treated as the probability of each tuple.

### Demand Transformation (Magic-Set Transformation)

Magic-set transformation allows some originally un-evaluable queries to be evaluable,
and can potentially optimize the program.
For example, a traditional fibonacci number program in Datalog will go to infinite.
However, with demand transformation we can evaluate this program:

```
@demand("bf")
def fib = {(0, 1), (1, 1)}
def fib(x, a + b) = fib(x - 1, a), fib(x - 2, b), x > 1
query fib(10, y)
```

Note that we have `@demand("bf")` specified on the `fib` relation.
Since we are just curious the ten-th number of fibonacci, this program is evaluable.
