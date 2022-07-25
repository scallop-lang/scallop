use sdd::*;

#[test]
fn left_linear_vtree() {
  let vtree_5 = VTree::create_left_linear(5);
  println!("{:?}", vtree_5);
}

#[test]
fn left_linear_vtree_1() {
  let vtree_1 = VTree::create_left_linear(1);
  println!("{:?}", vtree_1);
}

#[test]
fn balanced_vtree_5() {
  let vtree = VTree::create_balanced(5);
  println!("{:?}", vtree);
}

#[test]
fn vtree_mutation_1() {
  let mut vtree = VTree::create_balanced(5);
  vtree.save_dot("dots/before.dot").unwrap();
  vtree.swap(vtree.root_id()).unwrap();
  vtree.save_dot("dots/after.dot").unwrap();
}

#[test]
fn vtree_mutation_2() {
  let mut vtree = VTree::create_balanced(5);
  vtree.save_dot("dots/before.dot").unwrap();
  vtree.rotate_left(vtree.root_id()).unwrap();
  vtree.save_dot("dots/after.dot").unwrap();
}

#[test]
fn vtree_mutation_3() {
  let mut vtree = VTree::create_balanced(5);
  vtree.save_dot("dots/before.dot").unwrap();
  vtree.rotate_right(vtree.root_id()).unwrap();
  vtree.save_dot("dots/after.dot").unwrap();
}
