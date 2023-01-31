use scallop_core::compiler::compile_string_to_ram;
use scallop_core::runtime::dynamic;
use scallop_core::runtime::env;
use scallop_core::runtime::provenance::*;
use scallop_core::utils::RcFamily;

#[test]
fn incr_exec_1() {
  let mut ctx = unit::UnitProvenance::default();
  let mut runtime = env::RuntimeEnvironment::default();
  let mut exec_ctx = dynamic::DynamicExecutionContext::<_, RcFamily>::new();

  let r1 = compile_string_to_ram(
    r#"
    rel edge = {(0, 1), (1, 2)}
    rel path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)
    query path(y, 2)
  "#
    .to_string(),
  )
  .unwrap();

  exec_ctx
    .incremental_execute(r1, &mut runtime, &mut ctx)
    .expect("Runtime Error");

  let r2 = compile_string_to_ram(
    r#"
    rel edge = {(0, 1), (1, 2)}
    rel path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)
    query path(0, x)
  "#
    .to_string(),
  )
  .unwrap();

  exec_ctx
    .incremental_execute(r2, &mut runtime, &mut ctx)
    .expect("Runtime Error");
}
