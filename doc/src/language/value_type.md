# Values and Types

Scallop has a built-in set of basic value types, following Rust's naming convention.
From there, we have types such as `Symbol`, `DateTime`, `Entity`, and `Tensor`, which are special types to Scallop.

| Type | Description |
|------|-------------|
| `i8` | Signed-integer, 8-bit |
| `i16` | Signed-integer, 16-bit |
| `i32` | Signed-integer, 32-bit |
| `i64` | Signed-integer, 64-bit |
| `i128` | Signed-integer, 128-bit |
| `isize` | Signed size; its size is dependent on the system |
| `u8` | Unsigned-integer, 8-bit |
| `u16` | Unsigned-integer, 16-bit |
| `u32` | Unsigned-integer, 32-bit |
| `u64` | Unsigned-integer, 64-bit |
| `u128` | Unsigned-integer, 128-bit |
| `usize` | Unsigned size; its size is dependent on the system |
| `f32` | Floating-point number, 32-bit |
| `f64` | Floating-point number, 64-bit |
| `bool` | Boolean |
| `char` | Character |
| `String` | Variable-length string |
| `Symbol` | Symbol |
| `DateTime` | Date and time |
| `Duration` | Duration |
| `Entity` | Entity |
| `Tensor` | Tensor |

### Integers

Integers are the most basic data-type in Scallop.
If not specified, the default integer type that the system will pick is the `i32` (signed integer 32-bit) type:

``` scl
rel edge = {(0, 1), (1, 2)} // (i32, i32)
```

If an unsigned integer type is specified but a negative number is used in the declared facts, a type inference error will be raised.
We demonstrate this in the `sclrepl` environment:

```
scl> type my_edge(usize, usize)
scl> rel my_edge = {(-1, -5), (0, 3)}
[Error] cannot unify types `usize` and `signed integer`, where the first is declared here
  REPL:0 | type my_edge(usize, usize)
         |              ^^^^^
and the second is declared here
  REPL:1 | rel my_edge = {(-1, -5), (0, 3)}
         |                 ^^
```

Primitive operations that can be used along with integers are

- Comparators:
  - `==` (equality)
  - `!=` (inequality)
  - `>` (greater-than)
  - `>=` (greater-than-or-equal-to)
  - `<` (less-than)
  - `<=` (less-than-or-equal-to)
- Arithmetic operators:
  - `+` (plus)
  - `-` (minus/negate)
  - `*` (mult)
  - `/` (div)
  - `%` (mod)

All of the above operations need to operate on two integers of the same type.
For instance, you cannot compare an `i32` value with a `usize` value.

### Floating Point Numbers

Floating point numbers are supported in Scallop as well.
The following example shows the definition of student and their class grades:

``` scl
type student_grade(name: String, class: String, grade: f32)

rel student_grade = {
  ("alice", "cse 100", 95.2),
  ("bob", "cse 100", 90.8),
}
```

It is possible derive special floating points such as `inf` and `-inf`, though we cannot declare such values directly.
For the floating point that is `nan` (not-a-number), we will omit the whole fact from the database to maintain sanity.
Specifically, the derivation of `nan` is treated as a failure of foreign functions, which we explain in detail [here](foreign_functions.md).

All the basic operations that can work on integers would be able to work for floating point numbers as well.

### Boolean

Scallop allows the use of boolean values (`true` and `false`).

``` scl
type variable_assign(String, bool)
rel variable_assign = {("a", true), ("b", false)}
```

We support the following boolean operations:

- Comparisons
  - `==` (equality)
  - `!=` (inequality)
- Logical operations
  - `!` (unary negate)
  - `&&` (binary and)
  - `||` (binary or)
  - `^` (binary xor)

For example, we can have the following code

``` scl
rel result(a ^ b) = variable_assign("a", a) and variable_assign("b", b) // true
```

### Character

Scallop allows definition of characters such as `'a'`, `'*'`.
They are single-quoted, and can contain escaped characters such as `'\n'` (new-line) and `'\t'` (tab).

``` scl
type my_chars = {(0, 'h'), (1, 'e'), (2, 'l'), (3, 'l'), (4, 'o')}
```

Comparisons operations `==` and `!=` are available for characters.

### String

Scallop support variable length strings of the type `String`.
Strings are declared using the double quote (`"`), and can contain escaped characters such as `\n` and `\t`.

``` scl
rel greeting = {"Hello World"}
```

Strings can certainly be compared using `==` and `!=`.
The main ways for interacting with strings are through foreign functions such as `$string_length`, `$substring`, `$string_concat`, and etc.
Please refer to the [foreign functions section](foreign_functions.md) for more information.

### Symbols

Symbols are internally registered strings.
They are most commonly created through [loading from external files](loading_csv.md).
But they can still be specified using the `s`-quoted-string notation:

``` scl
rel symbols = {s"NAME", s"AGE", s"GENDER"}
```

### DateTime and Duration

`DateTime` and `Duration` are natively supported data structures by Scallop.
We commonly specify `DateTime` and `Duration` using their string form.
In the following example, we specify the `DateTime` values using the `t`-quoted-string notation (`t` represents time):

``` scl
rel event_dates = {("enroll", t"2020-01-01"), ("finish", t"2020-03-01")}
```

The dates will be all transformed into UTC time-zone.
When the date part is specified and the time is not specified, we will fill the time `00:00:00 UTC`.
When the time is specified but the date is not, we will use the current date when the program is invoked.
Any reasonable date-time format are acceptable, common ones include

- `t"2019-11-29 08:08:05-08"`
- `t"4/8/2014 22:05"`
- `t"September 17, 2012 10:09am"`
- `t"2014/04/2 03:00:51"`
- `t"2014年04月08日"`

`Duration`s can be specified using the `d`-quoted-string notation (`d` represents duration):

``` scl
rel event_durations = {("e1", d"12 days"), ("e2", d"15 days 20 seconds")}
```

The string can contain numbers followed by their units.
When specifying durations, the following units are accepted:

- nanoseconds (`n`)
- microseconds (`usecs`)
- milliseconds (`msecs`)
- seconds (`secs`)
- minutes (`m`)
- hours (`h`)
- days (`d`)
- weeks (`w`)
- months (`M`)
- years (`y`)

We can operate between `Duration` and `DateTime` using simple operations such as `+` and `-`:
- `DateTime + Duration ==> DateTime`
- `Duration + Duration ==> Duration`
- `DateTime - DateTime ==> Duration`
- `DateTime - Duration ==> DateTime`
- `Duration - Duration ==> Duration`

### Entity

Entity values are 64-bit unsigned integers created through hashing.
They are used to represent pointers of created entities.
They cannot be directly created.
Rather, they are managed by Scallop through the creation of entities.
For example,

``` scl
type List = Nil() | Cons(i32, List)
const MY_LIST = Cons(1, Cons(2, Nil()))
rel input_list(MY_LIST)
query input_list
```

The result is then

```
input_list: {(entity(0x4cd0d9e6652cdfc7))}
```

Please refer to [this section](adt_and_entity.md) for more informaiton on algebraic data types and entities.

## Type Conversions

In Scallop, types can be converted using the `as` operator.
For example, we can have

``` scl
rel numbers = {1, 2, 3, 4, 5}
rel num_str(n as String) = numbers(n)
```

to derive the `numbers` to be `{"1", "2", "3", "4", "5"}`.
In general, we can have all numbers castable to each other.
We also have every type being castable to `String`.
For converting `String` to other types, it undergoes a parsing process.
When the parsing does not go through, no result will be returned.
