use scallop_core::common::expr::*;
use scallop_core::runtime::dynamic::*;
use scallop_core::runtime::env::*;
use scallop_core::runtime::provenance::*;
use scallop_core::testing::*;
use scallop_core::utils::*;

#[test]
fn test_dynamic_exclusion_1() {
  let ctx = proofs::ProofsProvenance::<RcFamily>::default();
  let rt = RuntimeEnvironment::default();

  // Relations
  let mut source = DynamicRelation::<proofs::ProofsProvenance<RcFamily>>::new();
  let mut target = DynamicRelation::<proofs::ProofsProvenance<RcFamily>>::new();
  source.insert_untagged(&ctx, vec![(0,), (1,)]);

  // Untagged vec for exclusion
  let exc = vec![("red".to_string(),).into(), ("blue".to_string(),).into()];

  // Iterate until fixpoint
  let mut first_time = true;
  while source.changed(&ctx, rt.get_default_scheduler())
    || target.changed(&ctx, rt.get_default_scheduler())
    || first_time
  {
    target.insert_dataflow_recent(
      &ctx,
      &dataflow::DynamicDataflow::dynamic_relation(&source)
        .dynamic_exclusion(dataflow::DynamicDataflow::untagged_vec(&ctx, exc.clone()), &ctx, &rt)
        .project((Expr::access((0, 0)), Expr::access((1, 0))).into(), &rt),
      &rt,
    );
    first_time = false;
  }

  // Inspect the result
  expect_collection(
    &target.complete(&ctx),
    vec![
      (0, "red".to_string()),
      (0, "blue".to_string()),
      (1, "red".to_string()),
      (1, "blue".to_string()),
    ],
  );
}
