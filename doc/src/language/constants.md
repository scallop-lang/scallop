# Declaring Constants

We can declare constants and give it names.
The general syntax is the following:

``` scl
const NAME (: TYPE)? = CONSTANT
```

For example, we can define the value of `PI`:

``` scl
const PI = 3.1415926
```

Notice that here we have not specified the type of `PI`.
By default, a float value would resort to the place where the constant is used.
If we want to specify a non-default type, we can do

``` scl
const PI: f64 = 3.1415926
```

We can also declare multiple constants at a time:

``` scl
const LEFT = 0, UP = 1, RIGHT = 2, DOWN = 3
```

## Enum Types

We sometimes want to define enum types which contain constant variables.
Common examples include `RED`, `GREEN`, and `BLUE` under the `Color` type, and `LEFT`, `RIGHT`, `UP` under the `Action` type.
These can be achieved by defining enum types:

``` scl
type Color = RED | GREEN | BLUE
type Action = LEFT | UP | RIGHT | DOWN
```

Internally, the values such as `RED` and `UP` are unsigned integer constants.
If not specified, the values start from 0 and goes up 1 at a time.

For example, given the type definition above, `RED = 0`, `GREEN = 1`, and `BLUE = 2`.
For `Action`s, `LEFT = 0`, `UP = 1`, and etc.
Notice that even when `Color` and `Action` are different types, their values can overlap.

One can specify the values of these enum variants by attaching actual numbers to them.
In the following example, we have explicitly assigned three values to the colors.

``` scl
type Color = RED = 3 | GREEN = 5 | BLUE = 7
```

We can also just set a few of those:

``` scl
type Color = RED | GREEN = 10 | BLUE
```

In this case, `RED = 0`, `GREEN = 10`, and `BLUE = 11`.
Notice how blue's value is incremented from `GREEN`.

## Displaying Constants

Constants are just values and many of them are integer values.
They are not explicitly associated with any symbols.
If you want to display them correctly, we advise you create auxilliary relations storing the mapping from each constant to its string form.
For example, we can have

``` scl
rel color_to_string = {(RED, "red"), (GREEN, "green"), (BLUE, "blue")}
```

In this case, following the result with `color_to_string` relation will display their desired meanings properly.
