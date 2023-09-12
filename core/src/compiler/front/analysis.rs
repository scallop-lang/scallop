use crate::common::foreign_function::ForeignFunctionRegistry;
use crate::common::foreign_predicate::ForeignPredicateRegistry;

use super::analyzers::*;
use super::*;

/// The front analysis object that stores all the analysis results and errors
#[derive(Clone, Debug)]
pub struct Analysis {
  pub invalid_constant: InvalidConstantAnalyzer,
  pub invalid_wildcard: InvalidWildcardAnalyzer,
  pub input_files_analysis: InputFilesAnalysis,
  pub output_files_analysis: OutputFilesAnalysis,
  pub hidden_analysis: HiddenRelationAnalysis,
  pub aggregation_analysis: AggregationAnalysis,
  pub character_literal_analysis: CharacterLiteralAnalysis,
  pub constant_decl_analysis: ConstantDeclAnalysis,
  pub adt_analysis: AlgebraicDataTypeAnalysis,
  pub head_relation_analysis: HeadRelationAnalysis,
  pub type_inference: TypeInference,
  pub boundness_analysis: BoundnessAnalysis,
  pub demand_attr_analysis: DemandAttributeAnalysis,
}

impl Analysis {
  /// Create a new front IR analysis object
  pub fn new(function_registry: &ForeignFunctionRegistry, predicate_registry: &ForeignPredicateRegistry) -> Self {
    Self {
      invalid_constant: InvalidConstantAnalyzer::new(),
      invalid_wildcard: InvalidWildcardAnalyzer::new(),
      input_files_analysis: InputFilesAnalysis::new(),
      output_files_analysis: OutputFilesAnalysis::new(),
      hidden_analysis: HiddenRelationAnalysis::new(),
      aggregation_analysis: AggregationAnalysis::new(),
      character_literal_analysis: CharacterLiteralAnalysis::new(),
      constant_decl_analysis: ConstantDeclAnalysis::new(),
      adt_analysis: AlgebraicDataTypeAnalysis::new(),
      head_relation_analysis: HeadRelationAnalysis::new(predicate_registry),
      type_inference: TypeInference::new(function_registry, predicate_registry),
      boundness_analysis: BoundnessAnalysis::new(predicate_registry),
      demand_attr_analysis: DemandAttributeAnalysis::new(),
    }
  }

  pub fn perform_pre_transformation_analysis(&mut self, items: &Vec<Item>) {
    let mut analyzers = (
      &mut self.input_files_analysis,
      &mut self.hidden_analysis,
      &mut self.output_files_analysis,
      &mut self.aggregation_analysis,
      &mut self.character_literal_analysis,
      &mut self.constant_decl_analysis,
      &mut self.adt_analysis,
      &mut self.invalid_constant,
      &mut self.invalid_wildcard,
    );
    items.walk(&mut analyzers);
  }

  pub fn process_items(&mut self, items: &Vec<Item>) {
    // Prepare the type inference module
    self
      .type_inference
      .extend_constant_types(self.constant_decl_analysis.compute_typed_constants());

    // Create the analyzers and walk the items
    let mut analyzers = (
      &mut self.head_relation_analysis,
      &mut self.type_inference,
      &mut self.demand_attr_analysis,
      &mut self.boundness_analysis,
    );
    items.walk(&mut analyzers);
  }

  pub fn post_analysis(&mut self) {
    self.head_relation_analysis.compute_errors();
    self.type_inference.check_query_predicates();
    self.type_inference.infer_types();
    self.demand_attr_analysis.check_arity(&self.type_inference);
    self.boundness_analysis.check_boundness(&self.demand_attr_analysis);
  }

  pub fn dump_errors(&mut self, error_ctx: &mut FrontCompileError) {
    error_ctx.extend(&mut self.input_files_analysis.errors);
    error_ctx.extend(&mut self.invalid_constant.errors);
    error_ctx.extend(&mut self.invalid_wildcard.errors);
    error_ctx.extend(&mut self.aggregation_analysis.errors);
    error_ctx.extend(&mut self.character_literal_analysis.errors);
    error_ctx.extend(&mut self.constant_decl_analysis.errors);
    error_ctx.extend(&mut self.adt_analysis.errors);
    error_ctx.extend(&mut self.head_relation_analysis.errors);
    error_ctx.extend(&mut self.type_inference.errors);
    error_ctx.extend(&mut self.boundness_analysis.errors);
    error_ctx.extend(&mut self.demand_attr_analysis.errors);
  }
}
