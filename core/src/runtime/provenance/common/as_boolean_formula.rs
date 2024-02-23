use sdd::Semiring;

use super::Disjunctions;

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

  /// Can be used to build an SDD
  fn wmc_with_disjunctions<S, V>(&self, s: &S, v: &V, disj: &Disjunctions) -> S::Element
  where
    S: Semiring,
    V: Fn(&usize) -> S::Element,
  {
    let formula = self.as_boolean_formula();

    // Adding disjunction as part of formula
    let formula_with_disj = sdd::bf_disjunction(std::iter::once(formula).chain(disj.disjunctions.iter().map(|disj| {
      sdd::bf_conjunction(disj.facts.iter().map(|to_be_neg_fact_id| {
        sdd::bf_disjunction(disj.facts.iter().map(|fact_id| {
          if fact_id == to_be_neg_fact_id {
            sdd::bf_neg(fact_id.clone())
          } else {
            sdd::bf_pos(fact_id.clone())
          }
        }))
      }))
    })));

    let sdd_config = sdd::bottom_up::SDDBuilderConfig::with_formula(&formula_with_disj);
    let sdd_builder = sdd::bottom_up::SDDBuilder::with_config(sdd_config);
    let sdd = sdd_builder.build(&formula_with_disj);
    sdd.eval_t(v, s)
  }
}
