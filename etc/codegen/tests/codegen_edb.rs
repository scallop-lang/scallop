use scallop_core::common::tuple_type::TupleType;
use scallop_core::common::value_type::FromType;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

#[test]
fn codegen_edge_path_with_edb_1() {
  mod edge_path {
    use scallop_codegen::scallop;
    scallop! {
      rel edge = {(0, 1), (1, 2)}
      rel path(a, b) = edge(a, b)
      rel path(a, c) = path(a, b) and edge(b, c)
    }
  }

  let edb = edge_path::create_edb::<unit::UnitProvenance>();
  assert_eq!(
    edb.type_of("edge").unwrap(),
    <TupleType as FromType<(i32, i32)>>::from_type()
  );
  assert_eq!(
    edb.type_of("path").unwrap(),
    <TupleType as FromType<(i32, i32)>>::from_type()
  );
}

#[test]
fn codegen_edge_path_with_edb_2() {
  mod edge_path {
    use scallop_codegen::scallop;
    scallop! {
      type edge(a: usize, b: usize)
      rel path(a, b) = edge(a, b)
      rel path(a, c) = path(a, b) and edge(b, c)
    }
  }

  // Add things to edb
  let mut edb = edge_path::create_edb::<unit::UnitProvenance>();
  edb
    .add_facts("edge", vec![(0usize, 1usize), (1, 2), (2, 3)])
    .expect("Cannot add edge");

  // Run with edb
  let mut ctx = unit::UnitProvenance::default();
  let result = edge_path::run_with_edb(&mut ctx, edb);
  expect_static_output_collection(&result.path, vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
}
