use scallop_core::common::expr::*;
use scallop_core::common::foreign_aggregate::*;
use scallop_core::common::value_type::ValueType;
use scallop_core::compiler::ram::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

fn test_iteration_1<Prov>() -> DynamicCollection<Prov>
where
  Prov: Provenance + Default,
{
  let ctx = Prov::default();
  let rt = RuntimeEnvironment::default();
  let mut iter = DynamicIteration::<Prov>::new();

  // First create relations
  iter.create_dynamic_relation("edge");
  iter.create_dynamic_relation("_edge_rev");
  iter.create_dynamic_relation("path");

  // Insert EDB facts
  iter
    .get_dynamic_relation_unsafe("edge")
    .insert_untagged(&ctx, vec![(0, 1), (1, 2), (1, 3)]);

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
  iter.add_output_relation_with_default_storage("path");
  iter.add_output_relation_with_default_storage("edge");

  // Run the iteration
  let mut result = iter.run(&ctx, &rt);

  // Test the result
  expect_collection(&result["path"], vec![(0, 1), (1, 2), (0, 2), (1, 3), (0, 3)]);
  expect_collection(&result["edge"], vec![(0, 1), (1, 2), (1, 3)]);

  // Return path
  result.remove("path").unwrap()
}

#[test]
fn test_iteration_1_unit() {
  let _path = test_iteration_1::<unit::UnitProvenance>();
}

#[test]
fn test_iteration_1_natural() {
  let _path = test_iteration_1::<natural::NaturalProvenance>();
}

#[test]
fn test_iteration_1_boolean() {
  let _path = test_iteration_1::<boolean::BooleanProvenance>();
}

fn test_iteration_2<Prov>() -> DynamicCollection<Prov>
where
  Prov: Provenance + Default,
{
  let ctx = Prov::default();
  let rt = RuntimeEnvironment::default();

  let result_1 = {
    let mut strata_1 = DynamicIteration::<Prov>::new();
    strata_1.create_dynamic_relation("color");
    strata_1.create_dynamic_relation("_color_rev");
    strata_1.get_dynamic_relation_unsafe("color").insert_untagged(
      &ctx,
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
    strata_1.add_output_relation_with_default_storage("_color_rev");
    strata_1.run(&ctx, &rt)
  };

  let result_2 = {
    let mut strata_2 = DynamicIteration::<Prov>::new();
    strata_2.create_dynamic_relation("color_count");
    strata_2.add_input_dynamic_collection("_color_rev", result_1["_color_rev"].as_ref());
    strata_2.add_update_dataflow(
      "color_count",
      Dataflow::reduce(
        "count".to_string(),
        AggregateInfo::default().with_input_var_types(vec![ValueType::I32]),
        "_color_rev",
        ReduceGroupByType::Implicit,
      ),
    );
    strata_2.add_output_relation_with_default_storage("color_count");
    strata_2.run(&ctx, &rt)
  };

  let mut result_3 = {
    let mut strata_3 = DynamicIteration::<Prov>::new();
    strata_3.create_dynamic_relation("max_color_count");
    strata_3.add_input_dynamic_collection("color_count", result_2["color_count"].as_ref());
    strata_3.add_update_dataflow(
      "max_color_count",
      Dataflow::reduce(
        "max".to_string(),
        AggregateInfo::default()
          .with_arg_var_types(vec![ValueType::Str])
          .with_input_var_types(vec![ValueType::I32]),
        "color_count",
        ReduceGroupByType::None,
      ),
    );
    strata_3.add_output_relation_with_default_storage("max_color_count");
    strata_3.run(&ctx, &rt)
  };

  expect_collection(&result_3["max_color_count"], vec![("blue", 3usize)]);

  result_3.remove("max_color_count").unwrap()
}

#[test]
fn test_iteration_2_unit() {
  let _path = test_iteration_2::<unit::UnitProvenance>();
}

#[test]
fn test_iteration_2_natural() {
  let _path = test_iteration_2::<natural::NaturalProvenance>();
}

#[test]
fn test_iteration_2_boolean() {
  let _path = test_iteration_2::<boolean::BooleanProvenance>();
}
