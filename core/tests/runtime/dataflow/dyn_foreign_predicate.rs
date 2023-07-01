use scallop_core::common::expr::*;
use scallop_core::common::tuple::*;
use scallop_core::common::value::*;
use scallop_core::runtime::dynamic::dataflow::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;

#[test]
fn test_dyn_dataflow_free_range() {
  let runtime = RuntimeEnvironment::new_std();
  let ctx = unit::UnitProvenance::new();
  let df = DynamicDataflow::foreign_predicate_ground(
    "range#usize".to_string(),
    vec![Value::USize(1), Value::USize(5)],
    true,
    &ctx,
  );
  let batch = df.iter_recent(&runtime).next().unwrap().collect::<Vec<_>>();
  for i in 1..5 {
    match &batch[i - 1].tuple {
      Tuple::Tuple(vs) => {
        assert_eq!(vs.len(), 1);
        match vs[0] {
          Tuple::Value(Value::USize(x)) if x == i => {
            // Good
          }
          _ => assert!(false),
        }
      }
      _ => assert!(false),
    }
  }
}

#[test]
fn test_dyn_dataflow_soft_lt_1() {
  let runtime = RuntimeEnvironment::new_std();
  let ctx = min_max_prob::MinMaxProbProvenance::new();
  let source_df = vec![
    DynamicElement::new((1.0, -1.0), 1.0),
    DynamicElement::new((1.0, 1.0), 1.0),
    DynamicElement::new((1.0, 5.0), 1.0),
  ];
  let df = DynamicDataflow::vec(&source_df).foreign_predicate_constraint(
    "soft_lt#f64".to_string(),
    vec![Expr::access(0), Expr::access(1)],
    &ctx,
  );
  let batch = df.iter_recent(&runtime).next().unwrap().collect::<Vec<_>>();
  for elem in batch {
    let tup: (f64, f64) = elem.tuple.as_tuple();
    if tup.0 < tup.1 {
      assert!(elem.tag > 0.5);
    } else if tup.0 == tup.1 {
      assert!(elem.tag == 0.5);
    } else {
      assert!(elem.tag < 0.5);
    }
  }
}

#[test]
fn test_dyn_dataflow_join_range() {
  let runtime = RuntimeEnvironment::new_std();
  let ctx = unit::UnitProvenance::new();
  let data = vec![
    DynamicElement::new((1usize, 3usize), unit::Unit),
    DynamicElement::new((10usize, 10usize), unit::Unit),
    DynamicElement::new((100usize, 101usize), unit::Unit),
  ];
  let df = DynamicDataflow::vec(&data).foreign_predicate_join(
    "range#usize".to_string(),
    vec![Expr::access(0), Expr::access(1)],
    &ctx,
  );
  let batch = df.iter_recent(&runtime).next().unwrap().collect::<Vec<_>>();
  assert_eq!(
    AsTuple::<((usize, usize), (usize,))>::as_tuple(&batch[0].tuple),
    ((1usize, 3usize), (1usize,))
  );
  assert_eq!(
    AsTuple::<((usize, usize), (usize,))>::as_tuple(&batch[1].tuple),
    ((1usize, 3usize), (2usize,))
  );
  assert_eq!(
    AsTuple::<((usize, usize), (usize,))>::as_tuple(&batch[2].tuple),
    ((100usize, 101usize), (100usize,))
  );
}
