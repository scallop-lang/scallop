use scallop_core::runtime::provenance::*;

mod sum_2 {
  use scallop_codegen::scallop;
  scallop! {
    type digit_1(i8)
    type digit_2(i8)
    rel sum_2(a + b) = digit_1(a), digit_2(b)
    query sum_2
  }
}

fn main() {
  // First set the unit context
  let mut ctx = top_k_proofs::TopKProofsContext::new(3);

  // Then create an edb and populate facts inside of it
  let mut edb = sum_2::create_edb::<top_k_proofs::TopKProofsContext>();
  edb
    .add_annotated_disjunction("digit_1", vec![(0.9, (0,)), (0.01, (1,)), (0.01, (2,))])
    .unwrap();
  edb
    .add_annotated_disjunction("digit_2", vec![(0.01, (0,)), (0.01, (1,)), (0.98, (2,))])
    .unwrap();

  // Run with edb
  let result = sum_2::run_with_edb(&mut ctx, edb);

  // Check the results
  println!("sum_2: {:?}", result.sum_2);
}
