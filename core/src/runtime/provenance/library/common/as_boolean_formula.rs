use sdd::Semiring;

pub trait AsBooleanFormula {
  /// Implement as boolean formula
  fn as_boolean_formula(&self) -> sdd::BooleanFormula;

  /// Can be used to build an SDD
  fn wmc<S, V>(&self, s: &S, v: &V) -> S::Element
  where
    S: Semiring,
    V: Fn(&usize) -> S::Element,
  {
    let formula = self.as_boolean_formula();
    let sdd_config = sdd::bottom_up::SDDBuilderConfig::with_formula(&formula);
    let sdd_builder = sdd::bottom_up::SDDBuilder::with_config(sdd_config);
    let sdd = sdd_builder.build(&formula);
    sdd.eval_t(v, s)
  }
}
