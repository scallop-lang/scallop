# Scallop Compiler

The Scallop Compiler `sclc` can be used in two different ways, a library or executable.

The library is `sclc_core` and should be used the following way from Rust:

``` rs
use sclc_core::*;
create_executable(opt, compile_options, ram)?;
```

The executable `sclc` is invoked in command line.
It takes in a Scallop file (`.scl`) and generates an executable by default.
The executable can be invoked directly.

``` sh
$ sclc program.scl
$ ./program
```
