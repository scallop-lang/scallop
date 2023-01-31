use super::*;
use crate::compiler::front::*;

#[derive(Clone, Debug)]
pub enum BoundnessAnalysisError {
  UnboundVariable { var_name: String, var_loc: Loc },
  HeadExprUnbound { loc: Loc },
  ConstraintUnbound { loc: Loc },
  ReduceArgUnbound { loc: Loc },
}

impl FrontCompileErrorTrait for BoundnessAnalysisError {
  fn error_type(&self) -> FrontCompileErrorType {
    FrontCompileErrorType::Error
  }

  fn report(&self, src: &Sources) -> String {
    match self {
      Self::UnboundVariable { var_name, var_loc } => {
        format!("Unbound variable `{}` in the rule\n{}", var_name, var_loc.report(src))
      }
      Self::HeadExprUnbound { loc } => {
        format!("Argument of the head of a rule is unbounded\n{}", loc.report(src))
      }
      Self::ConstraintUnbound { loc } => {
        format!("Constraint unbound\n{}", loc.report(src))
      }
      Self::ReduceArgUnbound { loc } => {
        format!("The argument for the aggregation is unbounded\n{}", loc.report(src))
      }
    }
  }
}
