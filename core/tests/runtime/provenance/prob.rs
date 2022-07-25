use scallop_core::common::expr::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::provenance::*;

#[test]
fn test_simple_probability_count() {
  let mut ctx = min_max_prob::MinMaxProbContext::default();

  let result_1 = {
    let mut strata_1 = DynamicIteration::<min_max_prob::Prob>::new();
    strata_1.create_dynamic_relation("color");
    strata_1.create_dynamic_relation("_color_rev");
    strata_1.get_dynamic_relation_unsafe("color").insert_tagged(
      &mut ctx,
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
      Dataflow::dynamic_relation("color").project((Expr::access(1), Expr::access(0))),
    );
    strata_1.add_output_relation("_color_rev");
    strata_1.run(&ctx)
  };

  let result_2 = {
    let mut strata_2 = DynamicIteration::<min_max_prob::Prob>::new();
    strata_2.create_dynamic_relation("color_count");
    strata_2.add_input_dynamic_collection("_color_rev", &result_1["_color_rev"]);
    strata_2.add_update_dataflow(
      "color_count",
      Groups::keyed_groups("_color_rev").aggregate(DynamicAggregateOp::count(Expr::access(1))),
    );
    strata_2.add_output_relation("color_count");
    strata_2.run(&ctx)
  };

  println!("{:?}", result_2)
}

#[test]
fn test_min_max_prob_count_max() {
  let mut ctx = min_max_prob::MinMaxProbContext::default();

  let result_1 = {
    let mut strata_1 = DynamicIteration::<min_max_prob::Prob>::new();
    strata_1.create_dynamic_relation("color");
    strata_1.create_dynamic_relation("_color_rev");
    strata_1.get_dynamic_relation_unsafe("color").insert_tagged(
      &mut ctx,
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
      Dataflow::dynamic_relation("color").project((Expr::access(1), Expr::access(0))),
    );
    strata_1.add_output_relation("_color_rev");
    strata_1.run(&ctx)
  };

  let result_2 = {
    let mut strata_2 = DynamicIteration::<min_max_prob::Prob>::new();
    strata_2.create_dynamic_relation("color_count");
    strata_2.add_input_dynamic_collection("_color_rev", &result_1["_color_rev"]);
    strata_2.add_update_dataflow(
      "color_count",
      Groups::keyed_groups("_color_rev").aggregate(DynamicAggregateOp::count(Expr::access(1))),
    );
    strata_2.add_output_relation("color_count");
    strata_2.run(&ctx)
  };

  println!("{:?}", result_2);

  let result_3 = {
    let mut strata_3 = DynamicIteration::<min_max_prob::Prob>::new();
    strata_3.create_dynamic_relation("max_color");
    strata_3.add_input_dynamic_collection("color_count", &result_2["color_count"]);
    strata_3.add_update_dataflow(
      "max_color",
      Groups::single_collection("color_count").aggregate(DynamicAggregateOp::max(
        Some(Expr::access(0)),
        Expr::access(1),
      )),
    );
    strata_3.add_output_relation("max_color");
    strata_3.run(&ctx)
  };

  println!("{:?}", result_2["color_count"]);
  println!("{:?}", result_3["max_color"]);
}
