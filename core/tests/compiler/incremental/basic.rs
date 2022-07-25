use scallop_core::compiler::compile_string_to_ram;

#[test]
fn incr_1() {
  let r1 = compile_string_to_ram(
    r#"
    rel edge = {(0, 1), (1, 2)}
    rel path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)
    query path(y, 2)
  "#
    .to_string(),
  )
  .unwrap();
  let r2 = compile_string_to_ram(
    r#"
    rel edge = {(0, 1), (1, 2)}
    rel path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)
    query path(0, x)
  "#
    .to_string(),
  )
  .unwrap();
  let incr = r1.persistent_relations(&r2);
  assert!(incr.contains("edge"));
  assert!(incr.contains("path"));
}

#[test]
fn incr_2() {
  let r1 = compile_string_to_ram(
    r#"
    rel edge = {(0, 1), (1, 2)}
    rel path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)
    query path(y, 2)
  "#
    .to_string(),
  )
  .unwrap();
  let r2 = compile_string_to_ram(
    r#"
    rel edge = {(0, 1), (1, 2), (2, 3)}
    rel path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)
    query path(y, 2)
  "#
    .to_string(),
  )
  .unwrap();
  let incr = r1.persistent_relations(&r2);
  assert!(incr.is_empty());
}

#[test]
fn incr_3() {
  let r1 = compile_string_to_ram(
    r#"
    rel edge = {(0, 1), (1, 2)}
    rel path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)
    query path(y, 2)
  "#
    .to_string(),
  )
  .unwrap();
  let r2 = compile_string_to_ram(
    r#"
    rel edge = {(0, 1), (1, 2)}
    rel path(a, c) = edge(a, c) \/ edge(a, b) /\ path(b, c)
    query path(y, 2)
  "#
    .to_string(),
  )
  .unwrap();
  let incr = r1.persistent_relations(&r2);
  assert!(incr.len() == 1);
  assert!(incr.contains("edge"));
}

#[test]
fn incr_4() {
  let r1 = compile_string_to_ram(
    r#"
    rel R = {(0, 1), (1, 2)}
    rel Q(0, b) = R(0, b)
  "#
    .to_string(),
  )
  .unwrap();
  let r2 = compile_string_to_ram(
    r#"
    rel R = {(0, 1), (1, 2)}
    rel Q(1, b) = R(1, b)
  "#
    .to_string(),
  )
  .unwrap();
  let incr = r1.persistent_relations(&r2);
  assert!(incr.len() == 1);
  assert!(incr.contains("R"));
}

#[test]
fn incr_5() {
  let r1 = compile_string_to_ram(
    r#"
    rel edge = {(0, 1), (1, 2)}
    rel path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)
    query path
  "#
    .to_string(),
  )
  .unwrap();
  let r2 = compile_string_to_ram(
    r#"
    rel edge = {(0, 1), (1, 2)}
    rel path(a, c) = edge(a, c) \/ path(a, b) /\ edge(b, c)
    query path(1, _)
  "#
    .to_string(),
  )
  .unwrap();
  let incr = r1.persistent_relations(&r2);
  assert!(incr.contains("edge"));
  assert!(incr.contains("path"));
}
