use sdd::*;

#[test]
fn bool_formula_1() {
  let form = (bf(0) & bf(1) & bf(2)) | (bf(3) & bf(4));
  println!("{:?}", form);
}
