use std::convert::*;

use scallop_core::common::foreign_function::*;
use scallop_core::common::type_family::*;
use scallop_core::common::value::*;
use scallop_core::integrate;
use scallop_core::runtime::provenance;
use scallop_core::testing::*;
use scallop_core::utils::*;

#[derive(Clone)]
pub struct Fib;

impl ForeignFunction for Fib {
  fn name(&self) -> String {
    "fib".to_string()
  }

  fn num_generic_types(&self) -> usize {
    1
  }

  fn generic_type_family(&self, i: usize) -> TypeFamily {
    assert_eq!(i, 0);
    TypeFamily::Integer
  }

  fn num_static_arguments(&self) -> usize {
    1
  }

  fn static_argument_type(&self, i: usize) -> ForeignFunctionParameterType {
    assert_eq!(i, 0);
    ForeignFunctionParameterType::Generic(0)
  }

  fn return_type(&self) -> ForeignFunctionParameterType {
    ForeignFunctionParameterType::Generic(0)
  }

  fn execute(&self, args: Vec<Value>) -> Option<Value> {
    match args[0] {
      Value::I8(i) => fib(i).map(Value::I8),
      Value::I16(i) => fib(i).map(Value::I16),
      Value::I32(i) => fib(i).map(Value::I32),
      Value::I64(i) => fib(i).map(Value::I64),
      Value::I128(i) => fib(i).map(Value::I128),
      Value::ISize(i) => fib(i).map(Value::ISize),
      Value::U8(i) => fib(i).map(Value::U8),
      Value::U16(i) => fib(i).map(Value::U16),
      Value::U32(i) => fib(i).map(Value::U32),
      Value::U64(i) => fib(i).map(Value::U64),
      Value::U128(i) => fib(i).map(Value::U128),
      Value::USize(i) => fib(i).map(Value::USize),
      _ => None,
    }
  }
}

fn fib<T: Integer>(t: T) -> Option<T> {
  if t == T::zero() {
    Some(T::one())
  } else if t == T::one() {
    Some(T::one())
  } else {
    if t > T::one() {
      let length: usize = TryInto::try_into(t).ok()?;
      let mut storage = vec![T::one(); length];
      for i in 2..length {
        storage[i] = storage[i - 2] + storage[i - 1];
      }
      Some(storage[storage.len() - 1])
    } else {
      None
    }
  }
}

#[test]
fn test_fib_ff() {
  let prov_ctx = provenance::unit::UnitProvenance::default();
  let mut ctx = integrate::IntegrateContext::<_, RcFamily>::new(prov_ctx);

  // Source
  ctx.register_foreign_function(Fib).unwrap();
  ctx.add_relation("R(i32)").unwrap();
  ctx.add_rule(r#"S(x, $fib(x)) = R(x)"#).unwrap();

  // Facts
  ctx
    .edb()
    .add_facts("R", vec![(-10i32,), (0,), (3,), (5,), (8,)])
    .unwrap();

  // Execution
  ctx.run().unwrap();

  // Result
  expect_output_collection(
    "S",
    ctx.computed_relation_ref("S").unwrap(),
    vec![(0i32, 1i32), (3, 2), (5, 5), (8, 21)],
  );
}

#[test]
fn ff_string_length_1() {
  expect_interpret_result(
    r#"
      rel strings = {"hello", "world!"}
      rel lengths(x, $string_length(x)) = strings(x)
    "#,
    (
      "lengths",
      vec![("hello".to_string(), 5usize), ("world!".to_string(), 6)],
    ),
  );
}

#[test]
fn ff_string_length_2() {
  expect_interpret_result(
    r#"
      rel strings = {"hello", "world!"}
      rel lengths(x, y) = strings(x), y == $string_length(x)
    "#,
    (
      "lengths",
      vec![("hello".to_string(), 5usize), ("world!".to_string(), 6)],
    ),
  );
}

#[test]
fn ff_string_concat_2() {
  expect_interpret_result(
    r#"
      rel strings = {"hello", "world!"}
      rel cat(x) = strings(a), strings(b), a != b, x == $string_concat(a, " ", b)
    "#,
    (
      "cat",
      vec![("hello world!".to_string(),), ("world! hello".to_string(),)],
    ),
  );
}

#[test]
fn ff_hash_1() {
  expect_interpret_result(
    r#"
      rel result(x) = x == $hash(1, 3)
    "#,
    ("result", vec![(7198375873285174811u64,)]),
  );
}

#[test]
fn ff_hash_2() {
  expect_interpret_result(
    r#"
      rel result($hash(1, 3))
    "#,
    ("result", vec![(7198375873285174811u64,)]),
  );
}

#[test]
fn ff_abs_1() {
  expect_interpret_result(
    r#"
      rel my_rel = {-1, 3, 5, -6}
      rel abs_result($abs(x)) = my_rel(x)
    "#,
    ("abs_result", vec![(1i32,), (3,), (5,), (6,)]),
  );
}

#[test]
fn ff_abs_2() {
  expect_interpret_result(
    r#"
      rel my_rel = {-1.5, 3.3, 5.0, -6.5}
      rel abs_result($abs(x)) = my_rel(x)
    "#,
    ("abs_result", vec![(1.5f32,), (3.3,), (5.0,), (6.5,)]),
  );
}

#[test]
fn ff_substring_1() {
  expect_interpret_result(
    r#"
      rel my_rel = {"hello world!"}
      rel result($substring(x, 0, 5)) = my_rel(x)
    "#,
    ("result", vec![("hello".to_string(),)]),
  );
}

#[test]
fn ff_substring_2() {
  expect_interpret_result(
    r#"
      rel my_rel = {"hello world!"}
      rel result($substring(x, 6)) = my_rel(x)
    "#,
    ("result", vec![("world!".to_string(),)]),
  );
}

#[test]
fn ff_floor_1() {
  expect_interpret_result(
    r#"
      rel my_rel = {-1.5, 3.8, 5.0, -6.0}
      rel result($floor(x)) = my_rel(x)
    "#,
    ("result", vec![(-2.0f32,), (3.0,), (5.0,), (-6.0,)]),
  );
}

#[test]
fn ff_floor_2() {
  expect_interpret_result(
    r#"
      rel my_rel = {-1, 50, 12345, 0}
      rel result($floor(x)) = my_rel(x)
    "#,
    ("result", vec![(-1i32,), (50,), (12345,), (0,)]),
  );
}

#[test]
fn ff_ceil_1() {
  expect_interpret_result(
    r#"
      rel my_rel = {-1.5, 3.8, 5.0, -6.0}
      rel result($ceil(x)) = my_rel(x)
    "#,
    ("result", vec![(-1.0f32,), (4.0,), (5.0,), (-6.0,)]),
  );
}

#[test]
fn ff_ceil_2() {
  expect_interpret_result(
    r#"
      rel my_rel = {-1, 50, 12345, 0}
      rel result($ceil(x)) = my_rel(x)
    "#,
    ("result", vec![(-1i32,), (50,), (12345,), (0,)]),
  );
}

#[test]
fn ff_exp_1() {
  expect_interpret_result(
    r#"
      rel my_rel = {-1.5, 3.8, 0.0}
      rel result($exp(x)) = my_rel(x)
    "#,
    ("result", vec![((-1.5f32).exp(),), (3.8f32.exp(),), (1.0,)]),
  );
}

#[test]
fn ff_exp2_1() {
  expect_interpret_result(
    r#"
      rel my_rel = {-1.5, 3.8, 0.0}
      rel result($exp2(x)) = my_rel(x)
    "#,
    ("result", vec![((-1.5f32).exp2(),), (3.8f32.exp2(),), (1.0,)]),
  );
}

#[test]
fn ff_log_1() {
  expect_interpret_result(
    r#"
      rel my_rel = {12.5, 345.89, 1.0, -2.7}
      rel result($log(x)) = my_rel(x)
    "#,
    ("result", vec![(12.5f32.ln(),), (345.89f32.ln(),), (0.0,)]),
  );
}

#[test]
fn ff_log2_1() {
  expect_interpret_result(
    r#"
      rel my_rel = {12.5, 345.89, 1.0, -2.7}
      rel result($log2(x)) = my_rel(x)
    "#,
    ("result", vec![(12.5f32.log2(),), (345.89f32.log2(),), (0.0,)]),
  );
}

#[test]
fn ff_pow_1() {
  expect_interpret_result(
    r#"
    rel base = {2, 7}
    rel exp = {3, 10, 0}
    rel result(n) = base(x), exp(y), n == $pow(x, y)
    "#,
    (
      "result",
      vec![
        (2i32.pow(3),),
        (2i32.pow(10),),
        (1,),
        (7i32.pow(3),),
        (7i32.pow(10),),
        (1,),
      ],
    ),
  );
}

#[test]
fn ff_powf_1() {
  expect_interpret_result(
    r#"
      rel base = {1.5, 3.8}
      rel exp = {-2.5, 4.8, 0.0}
      rel result(n) = base(x), exp(y), n == $powf(x, y)
    "#,
    (
      "result",
      vec![
        (1.5f32.powf(-2.5),),
        (1.5f32.powf(4.8),),
        (1.0,),
        (3.8f32.powf(-2.5),),
        (3.8f32.powf(4.8),),
        (1.0,),
      ],
    ),
  );
}

#[test]
fn ff_acos_1() {
  expect_interpret_result(
    r#"
      rel my_rel = {-1.0, 1.0, 0.0, 1.5, -1.5}
      rel result($acos(x)) = my_rel(x)
    "#,
    ("result", vec![((-1.0f32).acos(),), (1.0f32.acos(),), (0.0f32.acos(),)]),
  );
}

#[test]
fn ff_asin_1() {
  expect_interpret_result(
    r#"
      rel my_rel = {-1.0, 1.0, 0.0, 1.5, -1.5}
      rel result($asin(x)) = my_rel(x)
    "#,
    ("result", vec![((-1.0f32).asin(),), (1.0f32.asin(),), (0.0f32.asin(),)]),
  );
}

#[test]
fn ff_atan_1() {
  expect_interpret_result(
    r#"
      rel my_rel = {-1.0, 1.0, 0.0}
      rel result($atan(x)) = my_rel(x)
    "#,
    ("result", vec![((-1.0f32).atan(),), (1.0f32.atan(),), (0.0f32.atan(),)]),
  );
}

#[test]
fn ff_atan2_1() {
  expect_interpret_result(
    r#"
      rel first = {1.5, -3.8, 0.0}
      rel second = {-2.5, 4.8, 0.0}
      rel result(n) = first(y), second(x), n == $atan2(y, x)
    "#,
    (
      "result",
      vec![
        (1.5f32.atan2(-2.5),),
        (1.5f32.atan2(4.8),),
        (1.5f32.atan2(0.0),),
        ((-3.8f32).atan2(-2.5),),
        ((-3.8f32).atan2(4.8),),
        ((-3.8f32).atan2(0.0),),
        (0.0f32.atan2(-2.5),),
        (0.0f32.atan2(4.8),),
        (0.0f32.atan2(0.0),),
      ],
    ),
  );
}

#[test]
fn ff_sign_1() {
  expect_interpret_result(
    r#"
      rel my_rel = {-12.5, 34.6, 0.0}
      rel result($sign(x)) = my_rel(x)
    "#,
    ("result", vec![(-1i32,), (1,), (0,)]),
  );
}

#[test]
fn ff_sign_2() {
  expect_interpret_result(
    r#"
      rel my_rel = {-12, 34, 0}
      rel result($sign(x)) = my_rel(x)
    "#,
    ("result", vec![(-1i32,), (1,), (0,)]),
  );
}

#[test]
fn ff_format_1() {
  expect_interpret_result(
    r#"
      rel strings = {"hello", "world!"}
      rel result(x) = strings(a), strings(b), a != b, x == $format("{} {}", a, b)
    "#,
    (
      "result",
      vec![("hello world!".to_string(),), ("world! hello".to_string(),)],
    ),
  );
}

#[test]
fn ff_format_2() {
  expect_interpret_result(
    r#"
      rel numbers = {1.2, -3.4}
      rel result(x) = numbers(a), numbers(b), a != b, x == $format("{} {}", a, b)
    "#,
    ("result", vec![("1.2 -3.4".to_string(),), ("-3.4 1.2".to_string(),)]),
  );
}

#[test]
fn ff_string_lower_1() {
  expect_interpret_result(
    r#"
      rel string = {"Hello World!", "aBcDeF1234."}
      rel result($string_lower(s)) = string(s)
    "#,
    (
      "result",
      vec![("hello world!".to_string(),), ("abcdef1234.".to_string(),)],
    ),
  );
}

#[test]
fn ff_string_upper_1() {
  expect_interpret_result(
    r#"
      rel string = {"Hello World!", "aBcDeF1234."}
      rel result($string_upper(s)) = string(s)
    "#,
    (
      "result",
      vec![("HELLO WORLD!".to_string(),), ("ABCDEF1234.".to_string(),)],
    ),
  );
}

#[test]
fn ff_string_index_of_1() {
  expect_interpret_result(
    r#"
      rel string = {"Scallop is cool!"}
      rel substring = {"o", "is", "Scallop"}
      rel result(i) = string(s), substring(t), i == $string_index_of(s, t)
    "#,
    ("result", vec![(5usize,), (8,), (0,)]),
  );
}

#[test]
fn ff_string_trim_1() {
  expect_interpret_result(
    r#"
      rel string = {"Hello World!", " \t ABC def 123 \n"}
      rel result($string_trim(s)) = string(s)
    "#,
    (
      "result",
      vec![("Hello World!".to_string(),), ("ABC def 123".to_string(),)],
    ),
  );
}
