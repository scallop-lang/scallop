use sdd::*;

#[test]
fn test_disjunction_1() {
  let form = ((bf(1) & bf(3)) | (bf(2) & bf(4))) & (!(bf(1) & bf(3)));
  let config = bottom_up::SDDBuilderConfig::with_formula(&form);
  let sdd = bottom_up::SDDBuilder::with_config(config).build(&form);
  sdd.save_dot("dots/disj_1.dot").unwrap();
  println!("{:?}", sdd);
}

#[test]
fn test_no_disjunction_1() {
  let form = (bf(1) & bf(3)) | (bf(2) & bf(4));
  let config = bottom_up::SDDBuilderConfig::with_formula(&form);
  let sdd = bottom_up::SDDBuilder::with_config(config).build(&form);
  sdd.save_dot("dots/disj_2.dot").unwrap();
  println!("{:?}", sdd);
}
