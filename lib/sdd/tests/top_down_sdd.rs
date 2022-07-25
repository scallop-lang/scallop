use sdd::*;

#[test]
fn test_td_sdd_7() {
  // (A ^ B) v (B ^ C) v (C ^ D)
  let form = (bf(0) & bf(1)) | (bf(1) & bf(2)) | (bf(2) & bf(3));
  let vars = vec![1, 0, 3, 2];
  let config = top_down::SDDBuilderConfig::new(vars, VTreeType::Right);
  let sdd = top_down::SDDBuilder::with_config(config).build(form);
  println!("{:?}", sdd);
  for a in &[false, true] {
    for b in &[false, true] {
      for c in &[false, true] {
        for d in &[false, true] {
          let expected = (a & b) | (b & c) | (c & d);
          let found = sdd.eval_i(vec![(0, *a), (1, *b), (2, *c), (3, *d)]);
          assert_eq!(
            expected, found,
            "Result is expected to be {} but is computed to be {}",
            expected, found
          );
        }
      }
    }
  }
}
