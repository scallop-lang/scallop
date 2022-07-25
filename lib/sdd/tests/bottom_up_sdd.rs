use sdd::*;

#[test]
fn test_bu_sdd_false() {
  let form = bf(0) & !bf(0);
  let config = bottom_up::SDDBuilderConfig::with_formula(&form);
  let sdd = bottom_up::SDDBuilder::with_config(config).build(&form);
  assert_eq!(sdd.eval_i(vec![(0, false)]), false);
  assert_eq!(sdd.eval_i(vec![(0, true)]), false);
}

#[test]
fn test_bu_sdd_true() {
  let form = bf(0) | !bf(0);
  let config = bottom_up::SDDBuilderConfig::with_formula(&form);
  let sdd = bottom_up::SDDBuilder::with_config(config).build(&form);
  assert_eq!(sdd.eval_i(vec![(0, false)]), true);
  assert_eq!(sdd.eval_i(vec![(0, true)]), true);
}

#[test]
fn test_bu_sdd_3() {
  let form = (bf(0) | !bf(1)) & bf(2);
  let config = bottom_up::SDDBuilderConfig::with_formula(&form);
  let sdd = bottom_up::SDDBuilder::with_config(config).build(&form);
  for a in &[true, false] {
    for b in &[true, false] {
      for c in &[true, false] {
        let expected = (a | (!b)) & c;
        let found = sdd.eval_i(vec![(0, *a), (1, *b), (2, *c)]);
        assert_eq!(expected, found);
      }
    }
  }
}

#[test]
fn test_bu_sdd_4() {
  let form = (bf(0) | !bf(1)) & bf(1);
  let config = bottom_up::SDDBuilderConfig::with_formula(&form);
  let sdd = bottom_up::SDDBuilder::with_config(config).build(&form);
  for a in &[true, false] {
    for b in &[true, false] {
      let expected = (a | (!b)) & b;
      let found = sdd.eval_i(vec![(0, *a), (1, *b)]);
      assert_eq!(expected, found);
    }
  }
}

#[test]
fn test_bu_sdd_5() {
  let form = (bf(0) & bf(1)) | (!bf(0) & !bf(1));
  let config = bottom_up::SDDBuilderConfig::with_formula(&form);
  let sdd = bottom_up::SDDBuilder::with_config(config).build(&form);
  for a in &[true, false] {
    for b in &[true, false] {
      let expected = (a & b) | (!a & !b);
      let found = sdd.eval_i(vec![(0, *a), (1, *b)]);
      assert_eq!(expected, found);
    }
  }
}

#[test]
fn test_bu_sdd_6() {
  // (A ^ B) v (B ^ C) v (C ^ D)
  let form = (bf(0) & bf(1)) | (bf(1) & bf(2)) | (bf(2) & bf(3));
  let config = bottom_up::SDDBuilderConfig::with_formula(&form);
  let sdd = bottom_up::SDDBuilder::with_config(config).build(&form);
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

/// This shows the Figure 1 of the paper
/// - SDD: A New Canonical Representation of Propositional Knowledge Bases
#[test]
fn test_bu_sdd_7() {
  // (A ^ B) v (B ^ C) v (C ^ D)
  let form = (bf(0) & bf(1)) | (bf(1) & bf(2)) | (bf(2) & bf(3));
  let vars = vec![1, 0, 3, 2];
  let config = bottom_up::SDDBuilderConfig::new(vars, VTreeType::Balanced, true);
  let sdd = bottom_up::SDDBuilder::with_config(config).build(&form);
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
