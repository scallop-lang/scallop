# Scallop Core

``` rust
fn main() {
  scallop_core::integrate::interpret_string(r#"
    rel edge = {(0, 1), (1, 2), (2, 3)}
    rel path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)
    query path(2, _)
  "#);
}
```

## Structure

Scallop core contains two main modules: `runtime` and `compiler`.
Compiler is used for compiling sources into executable programs.
Runtime is responsible for running such programs.
There is also an `integrate` module for people to use Scallop without
thinking about low level compilation or runtime details.
