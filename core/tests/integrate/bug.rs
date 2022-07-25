use scallop_core::testing::*;

#[test]
fn test_io_issue_3() {
  expect_interpret_empty_result(
    r#"
      rel c__ = {("lk", "tf")}
      rel d__(b, a) = c__(b, a), c__(h, h), c__(h, a)
      query d__
    "#,
    "d__",
  )
}

#[test]
fn test_io_issue_4() {
  expect_interpret_empty_result(
    r#"
      rel a__ = {
        ("ir", 38, 59),
        ("iz", 68, 32),
        ("as", 59, 69),
        ("ir", 49, 77),
      }
      rel b__(b, a) = a__(a, b, b), a__(a, b, b)
      query b__
    "#,
    "b__",
  )
}

#[test]
fn test_io_issue_18() {
  expect_interpret_empty_result(
    r#"
      rel e__ = {
        (49, 49, "ms"),
        (74, 74, "bx"),
      }

      rel h__(d, e, e) = e__(d, b, e), not e__(d, d, e)

      query h__
    "#,
    "h__",
  )
}
