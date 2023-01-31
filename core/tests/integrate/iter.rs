use scallop_core::testing::*;

#[test]
fn edge_path_iter_limit() {
  expect_interpret_within_iter_limit(
    r#"
    rel edge = {(0, 1), (1, 2), (2, 3), (3, 4)}
    rel path(a, c) = edge(a, c) or path(a, b) and edge(b, c)
    "#,
    8,
  )
}
