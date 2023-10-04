use scallop_core::integrate::*;

#[test]
fn test_uniform_sample() {
  let result = interpret_string(
    r#"
    rel numbers = {0, 1, 2, 3}
    rel sampled_number(x) = x := uniform<1>(x: numbers(x))
  "#
    .to_string(),
  )
  .expect("Failed executing");
  let sampled_number = result
    .get_output_collection("sampled_number")
    .expect("Cannot get `sampled_number` relation");
  assert_eq!(sampled_number.len(), 1, "There should be only one number being sampled");
  let number = sampled_number.ith_tuple(0).expect("There should be one number")[0].as_i32();
  assert!(number >= 0 && number < 4, "number should be anything between 0 and 3");
}
