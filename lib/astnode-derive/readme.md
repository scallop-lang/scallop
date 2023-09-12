# AstNode Macro Derive

This is a `proc-macro` derive package for the macro `AstNode`.
This macro is used to annotate Scallop's front-end AST nodes to save space on custom code.

AstNode can be annotated on three kinds of rust constructs

### 1. A struct

In this case, the structs should be named with an underscore `_` in front

``` rust
#[derive(AstNode)]
struct _MyNode {
  field1: T1,
  field2: Box<T2>, // with a box
  field3: Vec<T3>, // with a vec
  field4: Option<T4>, // with an option
}
```

The struct `_MyNode` should be annotated with `AstNode`.
When used, one should use the regular type without the underscore `MyNode`.
Each field can have types wrapped by `Box`, `Vec`, and `Option`, in which case constructors, getters, and setters are handled differently.
There is no restrictions on the field types.

You can create `MyNode` with a few constructors.
Please notice that for `field2` you don't need to wrap the type inside of `Box`.

``` rust
MyNode::new(field1: T1, field2: T2, field3: Vec<T3>, field4: Option<T4>)
MyNode::new_with_loc(field1: T1, field2: T2, field3: Vec<T3>, field4: Option<T4>, loc: NodeLocation)
MyNode::new_with_span(field1: T1, field2: T2, field3: Vec<T3>, field4: Option<T4>, start: usize, end: usize)
```

You can access each field using functions like `field1() -> &T1` or `field1_mut() -> &mut T1`.
For `Box` fields, the accessors will skip the `Box`: `field2() -> &T2`, `field2_mut() -> &mut T2`.
For `Vec` fields, there are functions like `iter_field3() -> Iterator`, `iter_field3_mut() -> Iterator`, `num_field3() -> usize`, and `has_field3> bool`.
For `Option` fields, there is a function `has_field4() -> bool`.

### 2. A variant enum

In this case, `AstNode` should annotate an enum that is named normally, with all variants having just a single argument

``` rust
#[derive(AstNode)]
enum EnumNode {
  V1(T1), // variant 1
  V2(T2), // variant 2
}
```

We assume that all types inside of the variants are (annotated by) `AstNode`.
As such, we have provided the following helper functions

``` rust
EnumNode::is_v1(&self) -> bool // there will be an `is_*` function for each variant
EnumNode::as_v1(&self) -> Option<&V1> // there will be an `as_*` function for each variant
```

### 3. A terminal enum

In this case, `AstNode` should annotate an enum that is named also with an underscore in front.
There should be at least one variant having no argument; all variants should not have more than one arguments.

``` rust
#[derive(AstNode)]
enum _TerminalNode {
  S1, // terminal variant
  S2, // terminal variant
  S3(T3), // non-terminal variant
}
```

All operations should be done on the normal type `TerminalNode`.
We offer the following helper functions

``` rust
TerminalNode::s1() -> Self // constructor without location
TerminalNode::s1_with_loc(loc: NodeLocation) // constructor with location provided
TerminalNode::s1_with_span(start: usize, end: usize) // constructor with location span provided
TerminalNode::s3(t3: T3) -> Self // providing the argument to construct variant s3
TerminalNode::s3_with_loc(t3: T3, loc: NodeLocation) -> Self // providing the argument to construct variant s3 with location provided
TerminalNode::s3_with_span(t3: T3, start: usize, end: usize) -> Self // providing the argument to construct variant s3 with location span provided
TerminalNode::is_s1(&self) -> bool // there is a `is_*` function for each variant
TerminalNode::as_s3(&self) -> Option<&T3> // there is an `as_*` function if that variant has an argument
```
