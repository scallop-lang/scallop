use scallop_core::common::expr::*;
use scallop_core::common::foreign_aggregate::*;
use scallop_core::common::value_type::ValueType;
use scallop_core::compiler::ram::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;

#[test]
fn test_simple_probability_count() {
  let ctx = min_max_prob::MinMaxProbProvenance::default();
  let rt = RuntimeEnvironment::default();

  let result_1 = {
    let mut strata_1 = DynamicIteration::<min_max_prob::MinMaxProbProvenance>::new();
    strata_1.create_dynamic_relation("color");
    strata_1.create_dynamic_relation("_color_rev");
    strata_1.get_dynamic_relation_unsafe("color").insert_tagged(
      &ctx,
      vec![
        (Some(0.5), (0usize, "blue")),
        (Some(0.8), (1, "green")),
        (Some(0.2), (2, "blue")),
        (Some(0.1), (3, "green")),
        (Some(0.1), (4, "blue")),
        (Some(0.2), (5, "red")),
      ],
    );
    strata_1.add_update_dataflow(
      "_color_rev",
      Dataflow::relation("color").project((Expr::access(1), Expr::access(0))),
    );
    strata_1.add_output_relation("_color_rev");
    strata_1.run(&ctx, &rt)
  };

  let result_2 = {
    let mut strata_2 = DynamicIteration::<min_max_prob::MinMaxProbProvenance>::new();
    strata_2.create_dynamic_relation("color_count");
    strata_2.add_input_dynamic_collection("_color_rev", &result_1["_color_rev"]);
    strata_2.add_update_dataflow(
      "color_count",
      Dataflow::reduce(
        "count".to_string(),
        AggregateInfo::default()
          .with_arg_var_types(vec![ValueType::Str])
          .with_input_var_types(vec![ValueType::USize]),
        "_color_rev",
        ReduceGroupByType::Implicit,
      ),
    );
    strata_2.add_output_relation("color_count");
    strata_2.run(&ctx, &rt)
  };

  println!("{:?}", result_2)
}

#[test]
fn test_min_max_prob_count_max() {
  let ctx = min_max_prob::MinMaxProbProvenance::default();
  let rt = RuntimeEnvironment::default();

  let result_1 = {
    let mut strata_1 = DynamicIteration::<min_max_prob::MinMaxProbProvenance>::new();
    strata_1.create_dynamic_relation("color");
    strata_1.create_dynamic_relation("_color_rev");
    strata_1.get_dynamic_relation_unsafe("color").insert_tagged(
      &ctx,
      vec![
        (Some(0.6), (0usize, "blue")),
        (Some(0.4), (0, "green")),
        (Some(0.3), (1, "blue")),
        (Some(0.7), (1, "green")),
        (Some(0.2), (2, "blue")),
        (Some(0.8), (2, "green")),
      ],
    );
    strata_1.add_update_dataflow(
      "_color_rev",
      Dataflow::relation("color").project((Expr::access(1), Expr::access(0))),
    );
    strata_1.add_output_relation("_color_rev");
    strata_1.run(&ctx, &rt)
  };

  let result_2 = {
    let mut strata_2 = DynamicIteration::<min_max_prob::MinMaxProbProvenance>::new();
    strata_2.create_dynamic_relation("color_count");
    strata_2.add_input_dynamic_collection("_color_rev", &result_1["_color_rev"]);
    strata_2.add_update_dataflow(
      "color_count",
      Dataflow::reduce(
        "count".to_string(),
        AggregateInfo::default().with_input_var_types(vec![ValueType::USize]),
        "_color_rev",
        ReduceGroupByType::Implicit,
      ),
    );
    strata_2.add_output_relation("color_count");
    strata_2.run(&ctx, &rt)
  };

  println!("{:?}", result_2);

  let result_3 = {
    let mut strata_3 = DynamicIteration::<min_max_prob::MinMaxProbProvenance>::new();
    strata_3.create_dynamic_relation("max_color");
    strata_3.add_input_dynamic_collection("color_count", &result_2["color_count"]);
    strata_3.add_update_dataflow(
      "max_color",
      Dataflow::reduce(
        "max".to_string(),
        AggregateInfo::default()
          .with_arg_var_types(vec![ValueType::Str])
          .with_input_var_types(vec![ValueType::USize]),
        "color_count",
        ReduceGroupByType::None,
      ),
    );
    strata_3.add_output_relation("max_color");
    strata_3.run(&ctx, &rt)
  };

  println!("{:?}", result_2["color_count"]);
  println!("{:?}", result_3["max_color"]);
}
