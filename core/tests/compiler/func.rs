use scallop_core::testing::*;

#[test]
fn func_simple() {
  expect_compile("type $add(x: i32, y: i32) -> i32")
}
