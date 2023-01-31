use std::convert::*;

use scallop_core::utils::*;
use scallop_core::common::value::*;
use scallop_core::common::foreign_function::*;
use scallop_core::common::type_family::*;
use scallop_core::runtime::provenance;
use scallop_core::integrate;
use scallop_core::testing::*;

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
  ctx.edb().add_facts("R", vec![(-10i32,), (0,), (3,), (5,), (8,)]).unwrap();

  // Execution
  ctx.run().unwrap();

  // Result
  expect_output_collection(
    ctx.computed_relation_ref("S").unwrap(),
    vec![(0i32, 1i32), (3, 2), (5, 5), (8, 21)],
  );
}
