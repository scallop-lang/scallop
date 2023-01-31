mod common;

#[test]
#[ignore]
fn pylib_edge_path() {
  common::check_compile_pylib_from_program_string(
    "edge_path_pylib_1",
    r#"
      rel edge = {(0, 1), (1, 2)}
      rel path(a, c) = edge(a, c) or (path(a, b) and edge(b, c))
    "#,
  )
}

#[test]
#[ignore]
fn exec_edge_path() {
  common::check_compile_exec_from_program_string(
    "edge_path_exec_1",
    r#"
      rel edge = {(0, 1), (1, 2)}
      rel path(a, c) = edge(a, c) or (path(a, b) and edge(b, c))
    "#,
  )
}
