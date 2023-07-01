use chrono::*;

use scallop_core::testing::*;

#[test]
fn date_type_1() {
  expect_compile(
    r#"
      type r(DateTime)
      rel r = {t"2019-01-01T00:00:00Z"}
    "#,
  )
}

#[test]
fn duration_type_1() {
  expect_compile(
    r#"
      type r(Duration)
      rel r = {d"1y5d"}
    "#,
  )
}

#[test]
fn date_1() {
  expect_interpret_result(
    r#"
      rel r = {t"2019-01-01T00:00:00Z"}
    "#,
    ("r", vec![(Utc.with_ymd_and_hms(2019, 01, 01, 0, 0, 0).unwrap(),)]),
  )
}

#[test]
fn date_2() {
  expect_interpret_result(
    r#"
      rel r = {(0, t"2019-01-01T00:00:00Z")}
    "#,
    ("r", vec![(0, Utc.with_ymd_and_hms(2019, 01, 01, 0, 0, 0).unwrap())]),
  )
}

#[test]
fn bad_date_1() {
  expect_front_compile_failure(r#"rel r = {t"ABCDEF"}"#, |e| {
    e.contains("Cannot parse date time `ABCDEF`")
  })
}

#[test]
fn bad_duration_1() {
  expect_front_compile_failure(r#"rel r = {d"ABCDEF"}"#, |e| {
    e.contains("Cannot parse duration `ABCDEF`")
  })
}

#[test]
fn date_plus_duration_1() {
  expect_interpret_result(
    r#"
      rel p = {t"2019-01-01T00:00:00Z"}
      rel q = {d"3d"}
      rel r(date + duration) = p(date) and q(duration)
    "#,
    ("r", vec![(Utc.with_ymd_and_hms(2019, 01, 04, 0, 0, 0).unwrap(),)]),
  )
}

#[test]
fn date_minus_duration_1() {
  expect_interpret_result(
    r#"
      rel p = {t"2019-01-04T00:00:00Z"}
      rel q = {d"3d"}
      rel r(date - duration) = p(date) and q(duration)
    "#,
    ("r", vec![(Utc.with_ymd_and_hms(2019, 01, 01, 0, 0, 0).unwrap(),)]),
  )
}

#[test]
fn duration_plus_duration_1() {
  expect_interpret_result(
    r#"
      rel p = {(d"3d", d"2d")}
      rel r(d1 + d2) = p(d1, d2)
    "#,
    ("r", vec![(Duration::days(5),)]),
  )
}

#[test]
fn get_year_1() {
  expect_interpret_result(
    r#"
      rel p = {t"2019-01-04T00:00:00Z"}
      rel r($datetime_year(d)) = p(d)
    "#,
    ("r", vec![(2019i32,)]),
  )
}

#[test]
fn get_month_1() {
  expect_interpret_result(
    r#"
      rel p = {t"2019-01-04T00:00:00Z"}
      rel r($datetime_month(d)) = p(d)
    "#,
    ("r", vec![(1u32,)]),
  )
}

#[test]
fn get_month0_1() {
  expect_interpret_result(
    r#"
      rel p = {t"2019-01-04T00:00:00Z"}
      rel r($datetime_month0(d)) = p(d)
    "#,
    ("r", vec![(0u32,)]),
  )
}
