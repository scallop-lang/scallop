# Installation

There are many ways in which you can use Scallop, forming a complete toolchain.
We specify how to installing the toolchain from source.
The following instructions assume you have access to the Scallop source code and have basic pre-requisites installed.

## Requirements

- Rust - nightly 2023-03-07 (please visit [here](https://rust-lang.github.io/rustup/concepts/channels.html) to learn more about Rust nightly and how to install them)
- Python 3.7+ (for connecting Scallop with Python and [PyTorch](https://pytorch.org))

## Scallop Interpreter

The interpreter of Scallop is named `scli`.
To install it, please do

``` bash
$ make install-scli
```

From here, you can use `scli` to test and run simple programs

``` bash
$ scli examples/datalog/edge_path.scl
```

## Scallop Interactive Shell


## Scallop Python Interface
