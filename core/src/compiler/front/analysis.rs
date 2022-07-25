use super::analyzers::*;
use super::*;

#[derive(Clone, Debug)]
pub struct Analysis {
  pub invalid_wildcard: InvalidWildcardAnalyzer,
  pub input_files_analysis: InputFilesAnalysis,
  pub output_files_analysis: OutputFilesAnalysis,
  pub hidden_analysis: HiddenRelationAnalysis,
  pub aggregation_analysis: AggregationAnalysis,
  pub type_inference: TypeInference,
  pub boundness_analysis: BoundnessAnalysis,
  pub demand_attr_analysis: DemandAttributeAnalysis,
}

impl Analysis {
  pub fn new() -> Self {
    Self {
      invalid_wildcard: InvalidWildcardAnalyzer::new(),
      input_files_analysis: InputFilesAnalysis::new(),
      output_files_analysis: OutputFilesAnalysis::new(),
      hidden_analysis: HiddenRelationAnalysis::new(),
      aggregation_analysis: AggregationAnalysis::new(),
      type_inference: TypeInference::new(),
      boundness_analysis: BoundnessAnalysis::new(),
      demand_attr_analysis: DemandAttributeAnalysis::new(),
    }
  }

  pub fn perform_pre_transformation_analysis(&mut self, items: &Vec<Item>) {
    let mut analyzers = (
      &mut self.input_files_analysis,
      &mut self.hidden_analysis,
      &mut self.output_files_analysis,
      &mut self.aggregation_analysis,
      &mut self.invalid_wildcard,
    );
    analyzers.walk_items(items);
  }

  pub fn process_items(&mut self, items: &Vec<Item>) {
    let mut analyzers = (
      &mut self.type_inference,
      &mut self.demand_attr_analysis,
      &mut self.boundness_analysis,
    );
    analyzers.walk_items(items);
  }

  pub fn post_analysis(&mut self) {
    self.type_inference.infer_types();
    self.demand_attr_analysis.check_arity(&self.type_inference);
    self
      .boundness_analysis
      .check_boundness(&self.demand_attr_analysis);
  }

  pub fn dump_errors(&mut self, error_ctx: &mut FrontErrorReportingContext) {
    error_ctx.extend(&mut self.input_files_analysis.errors);
    error_ctx.extend(&mut self.invalid_wildcard.errors);
    error_ctx.extend(&mut self.aggregation_analysis.errors);
    error_ctx.extend(&mut self.type_inference.errors);
    error_ctx.extend(&mut self.boundness_analysis.errors);
    error_ctx.extend(&mut self.demand_attr_analysis.errors);
  }
}
