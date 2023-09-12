use super::super::ast::*;
use super::super::error::*;
use super::super::source::*;
use super::super::utils::*;

#[derive(Debug, Clone)]
pub struct AggregationAnalysis {
  pub errors: Vec<AggregationAnalysisError>,
}

impl AggregationAnalysis {
  pub fn new() -> Self {
    Self { errors: vec![] }
  }
}

impl NodeVisitor<Reduce> for AggregationAnalysis {
  fn visit(&mut self, reduce: &Reduce) {
    // Check max/min arg
    match reduce.operator().internal() {
      _ReduceOp::Max | _ReduceOp::Min => {}
      _ReduceOp::Forall => {
        // Check the body of forall expression
        match reduce.body() {
          Formula::Implies(_) => {}
          _ => self.errors.push(AggregationAnalysisError::ForallBodyNotImplies {
            loc: reduce.location().clone(),
          }),
        }
      }
      _ReduceOp::Unknown(a) => self.errors.push(AggregationAnalysisError::UnknownAggregator {
        agg: a.clone(),
        loc: reduce.location().clone(),
      }),
      _ => {
        if !reduce.args().is_empty() {
          self
            .errors
            .push(AggregationAnalysisError::NonMinMaxAggregationHasArgument {
              op: reduce.operator().clone(),
            })
        }
      }
    }

    // Check the binding variables
    if reduce.bindings().is_empty() {
      match reduce.operator().internal() {
        _ReduceOp::Exists | _ReduceOp::Forall | _ReduceOp::Unknown(_) => {}
        _ => self.errors.push(AggregationAnalysisError::EmptyBinding {
          agg: reduce.operator().to_string(),
          loc: reduce.location().clone(),
        }),
      }
    }
  }
}

#[derive(Debug, Clone)]
pub enum AggregationAnalysisError {
  NonMinMaxAggregationHasArgument { op: ReduceOp },
  UnknownAggregator { agg: String, loc: Loc },
  ForallBodyNotImplies { loc: Loc },
  EmptyBinding { agg: String, loc: Loc },
}

impl FrontCompileErrorTrait for AggregationAnalysisError {
  fn error_type(&self) -> FrontCompileErrorType {
    FrontCompileErrorType::Error
  }

  fn report(&self, src: &Sources) -> String {
    match self {
      Self::NonMinMaxAggregationHasArgument { op } => {
        format!(
          "{} aggregation cannot have arguments\n{}",
          op.to_string(),
          op.location().report(src)
        )
      }
      Self::UnknownAggregator { agg, loc } => {
        format!("unknown aggregator `{}`\n{}", agg, loc.report(src))
      }
      Self::ForallBodyNotImplies { loc } => {
        format!(
          "the body of forall aggregation must be an `implies` formula\n{}",
          loc.report(src)
        )
      }
      Self::EmptyBinding { agg, loc } => {
        format!(
          "the binding variables of `{}` aggregation cannot be empty\n{}",
          agg,
          loc.report(src),
        )
      }
    }
  }
}
