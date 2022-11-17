pub mod aggregation;
pub mod boundness;
pub mod character_literal;
pub mod constant_decl;
pub mod demand_attr;
pub mod function;
pub mod head_relation;
pub mod hidden_relation;
pub mod input_files;
pub mod invalid_wildcard;
pub mod output_files;
pub mod type_inference;

pub use aggregation::AggregationAnalysis;
pub use boundness::BoundnessAnalysis;
pub use character_literal::CharacterLiteralAnalysis;
pub use constant_decl::ConstantDeclAnalysis;
pub use demand_attr::DemandAttributeAnalysis;
pub use function::FunctionAnalysis;
pub use head_relation::HeadRelationAnalysis;
pub use hidden_relation::HiddenRelationAnalysis;
pub use input_files::InputFilesAnalysis;
pub use invalid_wildcard::InvalidWildcardAnalyzer;
pub use output_files::OutputFilesAnalysis;
pub use type_inference::TypeInference;

pub mod errors {
  pub use super::aggregation::AggregationAnalysisError;
  pub use super::boundness::BoundnessAnalysisError;
  pub use super::constant_decl::ConstantDeclError;
  pub use super::demand_attr::DemandAttributeError;
  pub use super::function::FunctionAnalysisError;
  pub use super::head_relation::HeadRelationError;
  pub use super::input_files::InputFilesError;
  pub use super::invalid_wildcard::InvalidWildcardError;
  pub use super::output_files::OutputFilesError;
  pub use super::type_inference::TypeInferenceError;
}
