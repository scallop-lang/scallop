use rsat::*;

#[test]
fn rsat_variable() {
  let v = Variable::new(0);
  let l = Literal::positive(v);
  println!("{:?}", l);
}
