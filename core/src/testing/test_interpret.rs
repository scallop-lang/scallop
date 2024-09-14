use crate::common::tuple::Tuple;
use crate::compiler::*;
use crate::integrate::*;
use crate::runtime::database::*;
use crate::runtime::dynamic::*;
use crate::runtime::env::*;
use crate::runtime::error::*;
use crate::runtime::monitor;
use crate::runtime::provenance::*;
use crate::utils::*;

use super::*;

pub fn expect_interpret_result<T: Into<Tuple> + Clone>(s: &str, (p, e): (&str, Vec<T>)) {
  let actual = interpret_string(s.to_string()).expect("Compile Error");
  expect_output_collection(p, actual.get_output_collection_ref(p).unwrap(), e);
}

pub fn expect_interpret_result_with_runtime_option<T>(s: &str, o: RuntimeEnvironmentOptions, (p, e): (&str, Vec<T>))
where
  T: Into<Tuple> + Clone,
{
  let prov = unit::UnitProvenance::default();
  let opt = IntegrateOptions {
    compiler_options: CompileOptions::default(),
    execution_options: ExecutionOptions::default(),
    runtime_environment_options: o,
  };
  let mut interpret_ctx =
    InterpretContext::<_, RcFamily>::new_with_options(s.to_string(), prov, opt).expect("Compilation error");
  interpret_ctx.run().expect("Runtime error");
  let idb = interpret_ctx.idb();
  expect_output_collection(p, idb.get_output_collection_ref(p).unwrap(), e);
}

pub fn expect_interpret_result_with_setup<T, F>(s: &str, f: F, (p, e): (&str, Vec<T>))
where
  T: Into<Tuple> + Clone,
  F: FnOnce(&mut extensional::ExtensionalDatabase<unit::UnitProvenance>),
{
  let prov = unit::UnitProvenance::default();
  let mut interpret_ctx = InterpretContext::<_, RcFamily>::new(s.to_string(), prov).expect("Compilation error");
  f(interpret_ctx.edb());
  interpret_ctx.run().expect("Runtime error");
  let idb = interpret_ctx.idb();
  expect_output_collection(p, idb.get_output_collection_ref(p).unwrap(), e);
}

pub fn expect_interpret_result_with_tag<Prov, T, F>(s: &str, ctx: Prov, (p, e): (&str, Vec<(Prov::OutputTag, T)>), f: F)
where
  Prov: Provenance,
  T: Into<Tuple> + Clone,
  F: Fn(&Prov::OutputTag, &Prov::OutputTag) -> bool,
{
  let actual = interpret_string_with_ctx(s.to_string(), ctx).expect("Interpret Error");
  expect_output_collection_with_tag(p, actual.get_output_collection_ref(p).unwrap(), e, f);
}

/// Expect the given program to produce an empty relation `p`
///
/// ``` rust
/// # use scallop_core::testing::*;
/// expect_interpret_empty_result("type edge(i32, i32)", "edge")
/// ```
pub fn expect_interpret_empty_result(s: &str, p: &str) {
  let actual = interpret_string(s.to_string()).expect("Compile Error");
  assert!(
    actual.get_output_collection_ref(p).unwrap().is_empty(),
    "The relation `{}` is not empty",
    p
  )
}

/// Expect the given program to produce the expected relation/collections.
/// Panics if the program fails to compile/execute, or it does not produce the expected results.
pub fn expect_interpret_multi_result(s: &str, expected: Vec<(&str, TestCollection)>) {
  let actual = interpret_string(s.to_string()).expect("Compile Error");
  for (p, a) in expected {
    expect_output_collection(p, actual.get_output_collection_ref(p).unwrap(), a);
  }
}

/// Expect the given program to be executed within a given iteration limit.
/// It panics if the program uses an iteration count more than the limit.
pub fn expect_interpret_within_iter_limit(s: &str, iter_limit: usize) {
  let prov = unit::UnitProvenance::default();
  let monitor = monitor::IterationCheckingMonitor::new(iter_limit);
  interpret_string_with_ctx_and_monitor(s.to_string(), prov, &monitor).expect("Interpret Error");
}

/// Expect the given program to be executed within a given iteration limit.
/// It panics if the program uses an iteration count more than the limit.
pub fn expect_interpret_within_iter_limit_with_ctx_and_runtime_options<Prov: Provenance, Ptr: PointerFamily>(
  s: &str,
  iter_limit: usize,
  prov: Prov,
  runtime_environment_options: RuntimeEnvironmentOptions,
) {
  let monitor = monitor::IterationCheckingMonitor::new(iter_limit);
  let mut interpret_ctx = InterpretContext::<Prov, Ptr>::new_with_options(
    s.to_string(),
    prov,
    IntegrateOptions {
      compiler_options: CompileOptions::default(),
      execution_options: ExecutionOptions::default(),
      runtime_environment_options,
    },
  )
  .expect("Compile Error");
  interpret_ctx.run_with_monitor(&monitor).expect("Interpret Error");
}

pub fn expect_interpret_failure(s: &str) {
  let result = interpret_string(s.to_string());
  match result {
    Ok(_) => panic!("Interpreting succeeded instead of expected failure"),
    Err(err) => match err {
      IntegrateError::Compile(_) => panic!("Expecting runtime error but got compile error instead"),
      IntegrateError::Runtime(_) => { /* GOOD */ }
    },
  }
}

pub fn expect_interpret_specific_failure<F>(s: &str, f: F)
where
  F: Fn(RuntimeError) -> bool,
{
  let result = interpret_string(s.to_string());
  match result {
    Ok(_) => panic!("Interpreting succeeded instead of expected failure"),
    Err(err) => match err {
      IntegrateError::Compile(_) => panic!("Expecting runtime error but got compile error instead"),
      IntegrateError::Runtime(r) => {
        if f(r) {
          /* GOOD */
        } else {
          panic!("Did not capture expected runtime failure")
        }
      }
    },
  }
}
