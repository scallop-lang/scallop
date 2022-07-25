use super::*;
use crate::compiler::front::*;

#[derive(Clone, Debug)]
pub enum BoundnessAnalysisError {
  UnboundVariable { var_name: String, var_loc: Loc },
  HeadExprUnbound { loc: Loc },
  ConstraintUnbound { loc: Loc },
  ReduceArgUnbound { loc: Loc },
}

impl From<BoundnessAnalysisError> for FrontCompileError {
  fn from(e: BoundnessAnalysisError) -> Self {
    Self::BoundnessAnalysisError(e)
  }
}

impl BoundnessAnalysisError {
  pub fn report(&self, src: &Sources) {
    match self {
      Self::UnboundVariable { var_name, var_loc } => {
        println!("Unbound variable `{}` in the rule", var_name);
        var_loc.report(src);
      }
      Self::HeadExprUnbound { loc } => {
        println!("Argument of the head of a rule is unbounded");
        loc.report(src);
      }
      Self::ConstraintUnbound { loc } => {
        println!("Constraint unbound");
        loc.report(src);
      }
      Self::ReduceArgUnbound { loc } => {
        println!("The argument for the aggregation is unbounded");
        loc.report(src);
      }
    }
  }
}
