# Foreign Functions

Foreign functions allows for complex value manipulation in Scallop.
Conceptually, they are pure and partial functions that operate on value(s) and return one single value only.
Functions with states, such as `random`, are not allowed as foreign functions.

## Function Types

In Scallop, foreign functions are generically typed with optional and variable arguments.
All the functions have a dollar sign (`$`) associated with the function name.
We use the following syntax to denote a function signature

```
$FUNC_NAME<ARG(: FAMILY)?, ...>(
  POS_ARG: POS_ARG_TYPE, ...,
  OPT_ARG: OPT_ARG_TYPE?, ...,
  VAR_ARG_TYPE...
) -> RETURN_TYPE
```

The generic arguments are specified in the `<...>` after the function name, and can be annotated by optional type family.
For the arguments of the function, optional arguments have to appear after all positional arguments, and the variable arg type must appear after all positional and optional arguments.
Functions must have a return type.

For example, the function `$string_char_at(s: String, i: usize) -> char` takes in a string `s` and an index `i`, and returns the character at that location.
The two arguments `s` and `i` are both positional arguments.

In the function `$substring(s: String, b: usize, e: usize?)`, we have 2 positional arguments (`s` and `b`) and 1 optional argument (`e`).
This means that this substring function can be invoked with 2 or 3 arguments.
Invoking `$substring("hello", 3)` would give us `"lo"`, and invoking `$substring("hello", 1, 3)` would give us `"el"`.

For functions like `$abs<T: Number>(T) -> T`, we have absolute value function taking in value of any type that is a number (including integers and floating points).
The function also returns a type the same as the input.

For a function like `$format(f: String, Any...)`, it looks at the format string and fill all the `{}` symbol in the string with the latter arguments.
Notice how there can be arbitrary number of arguments (variable arg) of `Any` type.
For example, we can have `$format("{} + {}", 3, "a") ==> "3 + a"` and `$format("{}", true) ==> "true"`.

## Function Failures

Foreign functions may fail with errors such as divide-by-zero, index-out-of-bounds.
When error happens, values will not be propagated along the computation, and will be dropped silently.

For example, the following program makes use of the foreign function `$string_char_at`.
It walks through the indices 1, 3, and 5, and get the character on those indices of the string `"hello"`.

``` scl
rel indices = {1, 3, 5}
rel output(i, $string_char_at("hello", i)) = indices(i)
```

However, there are only 5 characters in the string, meaning that getting the `5`-th character would result in an index-out-of-bounds error.
Scallop will drop this invokation silently, resulting in only two facts being derived:

```
output: {(1, 'e'), (3, 'l')}
```

Similar things happen when `nan` is derived from floating point operations, or that the foreign function fails.

## Library of Foreign Functions

We hereby list all the available foreign functions, their signatures, descriptions, and an example of how to invoke them.
The functions here are ordered alphabetically.
For some of the functions that are slightly more complicated (e.g. `$format`), please refer to the section below for more information.

| Function | Description | Example |
|:---------|----------------------------------|-----------------|
| `$abs<T: Number>(x: T) -> T` | Absolute value function \\(\lvert x \rvert\\) | `$abs(-1)` => `1` |
| `$acos<T: Float>(x: T) -> T` | Arc cosine function \\(\text{acos}(x)\\) | `$acos(0.0)` => `1.5708` |
| `$atan<T: Float>(x: T) -> T` | Arc tangent function \\(\text{atan}(x)\\) | `$atan(0.0)` => `0.0` |
| `$atan2<T: Float>(y: T, x: T) -> T` | 2-argument arc tangent function \\( \text{atan}(y, x) \\) | `$atan2(0.0, 1.0)` => `0.0` |
| `$ceil<T: Number>(x: T) -> T` | Round *up* to closest integer \\( \lceil x \rceil \\) | `$ceil(0.5)` => `1.0` |
| `$cos<T: Float>(x: T) -> T` | Cosine function \\(\text{cos}(x)\\) | `$cos(0.0)` => `1.0` |
| `$datetime_day(d: DateTime) -> u32` | Get the day component of a `DateTime`, starting from 1 | `$datetime_day(t"2023-01-01")` => `1` |
| `$datetime_month(d: DateTime) -> u32` | Get the month component of a `DateTime`, starting from 1 | `$datetime_month(t"2023-01-01")` => `1` |
| `$datetime_month0(d: DateTime) -> u32` | Get the month component of a `DateTime`, starting from 0 | `$datetime_month0(t"2023-01-01")` => `0` |
| `$datetime_year(d: DateTime) -> i32` | Get the year component of a `DateTime` | `$datetime_month0(t"2023-01-01")` => `2023` |
| `$dot(a: Tensor, b: Tensor) -> Tensor` | Dot product of two tensors \\(a \cdot b\\); only available when compiled with `torch-tensor` |  |
| `$exp<T: Float>(x: T) -> T` | Exponential function \\(e^x\\) | `$exp(0.0)` => `1.0` |
| `$exp2<T: Float>(x: T) -> T` | Exponential function \\(2^x\\) (base 2) | `$exp2(2.0)` => `4.0` |
| `$floor<T: Number>(x: T) -> T` | Round *down* to closest integer \\(\lfloor x \rfloor\\) | `$exp2(2)` => `4` |
| `$format(String, Any...) -> String` | Formatting string | `$format("{} + {}", 3, "a")` => `"3 + a"` |
| `$hash(Any...) -> u64` | Hash the given values | `$hash("a", 3, 5.5)` => `5862532063111067262` |
| `$log<T: Float>(x: T) -> T` | Natural logarithm function \\(\text{log}_e(x)\\) | `$log(1.0)` => `0.0` |
| `$log2<T: Float>(x: T) -> T` | Logarithm function \\(\text{log}_2(x)\\) (base 2) | `$log2(4.0)` => `2.0` |
| `$max<T: Number>(T...) -> T` | Maximum \\(\text{max}(x_1, x_2, \dots)\\) | `$max(4.0, 1.0, 9.5)` => `9.5` |
| `$min<T: Number>(T...) -> T` | Minimum \\(\text{min}(x_1, x_2, \dots)\\) | `$max(4.0, 1.0, 9.5)` => `1.0` |
| `$pow<T: Integer>(x: T, y: u32) -> T` | Integer power function \\(x^y\\) | `$pow(2.2, 2)` => `4.84` |
| `$powf<T: Float>(x: T, y: T) -> T` | Float power function \\(x^y\\) | `$powf(4.0, 0.5)` => `2.0` |
| `$sign<T: Number>(x: T) -> T` | Sign function that returns \\(\{-1, 0, 1\}\\) in respective types | `$sign(-3.0)` => `-1.0` |
| `$sin<T: Float>(x: T) -> T` | Sine function \\(\text{sin}(x)\\) | `$sin(0.0)` => `0.0` |
| `$string_char_at(s: String, i: usize) -> char` | Get the `i`-th character of string `s` | `$string_char_at("hello", 2)` => `'l'` |
| `$string_concat(String...) -> String` | Concatenate multiple strings | `$string_concat("hello", " ", "world")` => `"hello world"` |
| `$string_index_of(s: String, pat: String) -> usize` | Find the index of the first occurrence of the pattern `pat` in string `s` | `$string_index_of("hello world", "world")` => `6` |
| `$string_length(s: String) -> usize` | Get the length of the string | `$string_length("hello")` => `5` |
| `$string_lower(s: String) -> String` | To lower-case | `$string_lower("LisA")` => `"lisa"` |
| `$string_trim(s: String) -> String` | Trim a string | `$string_trim("  hello ")` => `"hello"` |
| `$string_upper(s: String) -> String` | To upper-case | `$string_upper("LisA")` => `"LISA"` |
| `$substring(s: String, b: usize, e: usize?) -> String` | Find the substring given begin index and optional the end index | `$substring("hello world", 6)` => `"world"` |
| `$tan<T: Number>(x: T) -> T` | Tangent function \\(\text{tan}(x)\\) | `$tan(0.0)` => `0.0` |
