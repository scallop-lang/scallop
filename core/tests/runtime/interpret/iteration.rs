use scallop_core::common::aggregate_op::AggregateOp;
use scallop_core::common::expr::*;
use scallop_core::compiler::ram::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

fn test_iteration_1<T>() -> DynamicCollection<T>
where
  T: Tag + std::fmt::Debug,
  T::Context: ProvenanceContext + Default,
{
  let mut ctx = T::Context::default();
  let mut iter = DynamicIteration::<T>::new();

  // First create relations
  iter.create_dynamic_relation("edge");
  iter.create_dynamic_relation("_edge_rev");
  iter.create_dynamic_relation("path");

  // Insert EDB facts
  iter
    .get_dynamic_relation_unsafe("edge")
    .insert_untagged(&mut ctx, vec![(0, 1), (1, 2), (1, 3)]);

  // Insert updates
  iter.add_update_dataflow("path", Dataflow::relation("edge"));
  iter.add_update_dataflow(
    "_edge_rev",
    Dataflow::relation("edge").project((Expr::access(1), Expr::access(0))),
  );
  iter.add_update_dataflow(
    "path",
    Dataflow::relation("_edge_rev")
      .join(Dataflow::relation("path"))
      .project((Expr::access(1), Expr::access(2))),
  );

  // Specify output relations
  iter.add_output_relation("path");
  iter.add_output_relation("edge");

  // Run the iteration
  let mut result = iter.run(&ctx);

  // Test the result
  expect_collection(&result["path"], vec![(0, 1), (1, 2), (0, 2), (1, 3), (0, 3)]);
  expect_collection(&result["edge"], vec![(0, 1), (1, 2), (1, 3)]);

  // Return path
  result.remove("path").unwrap()
}

#[test]
fn test_iteration_1_unit() {
  let _path = test_iteration_1::<unit::Unit>();
}

#[test]
fn test_iteration_1_natural() {
  let _path = test_iteration_1::<natural::Natural>();
}

#[test]
fn test_iteration_1_boolean() {
  let _path = test_iteration_1::<boolean::Boolean>();
}

fn test_iteration_2<T>() -> DynamicCollection<T>
where
  T: Tag + std::fmt::Debug,
  T::Context: ProvenanceContext + Default,
{
  let mut ctx = T::Context::default();

  let result_1 = {
    let mut strata_1 = DynamicIteration::<T>::new();
    strata_1.create_dynamic_relation("color");
    strata_1.create_dynamic_relation("_color_rev");
    strata_1.get_dynamic_relation_unsafe("color").insert_untagged(
      &mut ctx,
      vec![
        (0, "blue"),
        (1, "green"),
        (2, "blue"),
        (3, "green"),
        (4, "blue"),
        (5, "red"),
      ],
    );
    strata_1.add_update_dataflow(
      "_color_rev",
      Dataflow::relation("color").project((Expr::access(1), Expr::access(0))),
    );
    strata_1.add_output_relation("_color_rev");
    strata_1.run(&ctx)
  };

  let result_2 = {
    let mut strata_2 = DynamicIteration::<T>::new();
    strata_2.create_dynamic_relation("color_count");
    strata_2.add_input_dynamic_collection("_color_rev", &result_1["_color_rev"]);
    strata_2.add_update_dataflow(
      "color_count",
      Dataflow::reduce(AggregateOp::Count, "_color_rev", ReduceGroupByType::Implicit),
    );
    strata_2.add_output_relation("color_count");
    strata_2.run(&ctx)
  };

  let mut result_3 = {
    let mut strata_3 = DynamicIteration::<T>::new();
    strata_3.create_dynamic_relation("max_color_count");
    strata_3.add_input_dynamic_collection("color_count", &result_2["color_count"]);
    strata_3.add_update_dataflow(
      "max_color_count",
      Dataflow::reduce(AggregateOp::Argmax, "color_count", ReduceGroupByType::None),
    );
    strata_3.add_output_relation("max_color_count");
    strata_3.run(&ctx)
  };

  expect_collection(&result_3["max_color_count"], vec![("blue", 3usize)]);

  result_3.remove("max_color_count").unwrap()
}

#[test]
fn test_iteration_2_unit() {
  let _path = test_iteration_2::<unit::Unit>();
}

#[test]
fn test_iteration_2_natural() {
  let _path = test_iteration_2::<natural::Natural>();
}

#[test]
fn test_iteration_2_boolean() {
  let _path = test_iteration_2::<boolean::Boolean>();
}
