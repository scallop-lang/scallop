use scallop_core::runtime::provenance::*;

mod edge_path {
  use scallop_codegen::scallop;
  scallop! {
    type edge(usize, usize)
    rel path(a, b) = edge(a, b)
    rel path(a, c) = path(a, b) and edge(b, c)
  }
}

fn main() {
  // First set the unit context
  let mut ctx = unit::UnitProvenance::default();

  // Then create an edb and populate facts inside of it
  let mut edb = edge_path::create_edb::<unit::UnitProvenance>();
  edb
    .add_facts("edge", vec![(0usize, 1usize), (1, 2), (2, 3), (3, 4)])
    .unwrap();

  // Run with edb
  let result = edge_path::run_with_edb(&mut ctx, edb);

  // Check the results
  println!("edge: {:?}", result.edge);
  println!("path: {:?}", result.path);
}
