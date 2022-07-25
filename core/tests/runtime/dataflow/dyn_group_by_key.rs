use scallop_core::common::expr::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;

fn test_group_by_key_1<T>() -> DynamicCollection<T>
where
  T: Tag + std::fmt::Debug,
  T::Context: ProvenanceContext<InputTag = ()> + Default,
{
  let mut ctx = T::Context::default();

  let result_1 = {
    let mut strata_1 = DynamicIteration::<T>::new();
    strata_1.create_dynamic_relation("color");
    strata_1.create_dynamic_relation("colors");
    strata_1.create_dynamic_relation("_color_rev");
    strata_1.create_dynamic_relation("_colors_key");
    strata_1
      .get_dynamic_relation_unsafe("color")
      .insert_untagged(&mut ctx, vec![(0usize, "blue"), (1, "green"), (2, "blue")]);
    strata_1
      .get_dynamic_relation_unsafe("colors")
      .insert_untagged(&mut ctx, vec![("blue",), ("green",), ("red",)]);
    strata_1.add_update_dataflow(
      "_color_rev",
      Dataflow::dynamic_relation("color").project((Expr::access(1), Expr::access(0))),
    );
    strata_1.add_update_dataflow(
      "_colors_key",
      Dataflow::dynamic_relation("colors").project((Expr::access(0), ())),
    );
    strata_1.add_output_relation("_color_rev");
    strata_1.add_output_relation("_colors_key");
    strata_1.run(&ctx)
  };

  let mut result_2 = {
    let mut strata_2 = DynamicIteration::<T>::new();
    strata_2.create_dynamic_relation("color_count");
    strata_2.add_input_dynamic_collection("_color_rev", &result_1["_color_rev"]);
    strata_2.add_input_dynamic_collection("_colors_key", &result_1["_colors_key"]);
    strata_2.add_update_dataflow(
      "color_count",
      Groups::group_by_join_collection("_colors_key", "_color_rev")
        .aggregate(DynamicAggregateOp::count(Expr::access(1)))
        .project((Expr::access(0), Expr::access(2))),
    );
    strata_2.add_output_relation("color_count");
    strata_2.run(&ctx)
  };

  result_2.remove("color_count").unwrap()
}

#[test]
fn test_group_by_key_1_unit() {
  let color_count = test_group_by_key_1::<unit::Unit>();
  expect_collection(
    &color_count,
    vec![("blue", 2usize), ("green", 1usize), ("red", 0usize)],
  )
}
