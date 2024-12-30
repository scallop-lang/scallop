# `scallopy`, the Python binding for Scallop

## Quick Start

This can be quickly installed using the following command through Github

``` bash
pip install "git+https://github.com/scallop-lang/scallop.git#egg=scallopy&subdirectory=etc/scallopy"
```

## Sample usage

``` python
from scallopy import ScallopContext

# Create new context
ctx = ScallopContext()

# Construct the program
ctx.add_relation("edge", (int, int))
ctx.add_facts("edge", [(0, 1), (1, 2), (2, 3)])
ctx.add_rule("path(a, c) = edge(a, c)")
ctx.add_rule("path(a, c) = edge(a, b), path(b, c)")

# Run the program
ctx.run()

# Inspect the result
paths = ctx.relation("path")
for (s, t) in paths:
  print(f"elem: ({s}, {t})")
```

## Build and Use

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
$ cd etc/scallopy # Go to this directory
$ make            # Build the library
$ cd ../..        # Going back to the root `scallop` directory
```

If succeed, please run some examples just to confirm that `scallopy` is indeed installed successfully.
When doing so (and all of the above), please make sure that you are inside of the virtual environment or
conda environment.

``` bash
$ python examples/edge_path.py
```

If you want to run the experiments in `/experiments` folder, you should additionally install PyTorch,
PyTorch Vision, and `tqdm`.

``` bash
$ pip install tqdm torch torchvision
$ python experiments/mnist/sum_2.py
```

## Documentation

Please check out the following file [`scallopy/scallopy.pyi`](scallopy/scallopy.pyi) for documentation.
