use super::super::ast::*;
use super::super::error::*;
use super::super::source::*;
use super::super::utils::*;
use super::super::visitor::*;

#[derive(Debug, Clone)]
pub struct AggregationAnalysis {
  pub errors: Vec<AggregationAnalysisError>,
}

impl AggregationAnalysis {
  pub fn new() -> Self {
    Self { errors: vec![] }
  }
}

impl NodeVisitor for AggregationAnalysis {
  fn visit_reduce(&mut self, reduce: &Reduce) {
    // Check max/min arg
    match &reduce.operator().node {
      ReduceOperatorNode::Max | ReduceOperatorNode::Min => {}
      ReduceOperatorNode::Forall => {
        // Check the body of forall expression
        match reduce.body() {
          Formula::Implies(_) => {}
          _ => self.errors.push(AggregationAnalysisError::ForallBodyNotImplies {
            loc: reduce.location().clone(),
          }),
        }
      }
      ReduceOperatorNode::Unknown(a) => self.errors.push(AggregationAnalysisError::UnknownAggregator {
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
  }
}

#[derive(Debug, Clone)]
pub enum AggregationAnalysisError {
  NonMinMaxAggregationHasArgument { op: ReduceOperator },
  UnknownAggregator { agg: String, loc: Loc },
  ForallBodyNotImplies { loc: Loc },
}

impl AggregationAnalysisError {
  pub fn report(&self, src: &Sources) {
    match self {
      Self::NonMinMaxAggregationHasArgument { op } => {
        println!("{} aggregation cannot have arguments", op);
        op.location().report(src);
      }
      Self::UnknownAggregator { agg, loc } => {
        println!("unknown aggregator `{}`", agg);
        loc.report(src);
      }
      Self::ForallBodyNotImplies { loc } => {
        println!("the body of forall aggregation must be an `implies` formula");
        loc.report(src);
      }
    }
  }
}

impl From<AggregationAnalysisError> for FrontCompileError {
  fn from(e: AggregationAnalysisError) -> Self {
    Self::AggregationAnalysisError(e)
  }
}
