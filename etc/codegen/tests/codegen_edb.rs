use scallop_core::common::tuple_type::TupleType;
use scallop_core::common::value_type::FromType;
use scallop_core::runtime::provenance::*;

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

  let edb = edge_path::create_edb::<unit::UnitContext>();
  assert_eq!(
    edb.type_of("edge").unwrap(),
    <TupleType as FromType<(usize, usize)>>::from_type()
  );
  assert_eq!(
    edb.type_of("path").unwrap(),
    <TupleType as FromType<(usize, usize)>>::from_type()
  );
}
