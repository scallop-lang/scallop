pub mod aggregation;
pub mod boundness;
pub mod demand_attr;
pub mod hidden_relation;
pub mod input_files;
pub mod invalid_wildcard;
pub mod output_files;
pub mod type_inference;

pub use aggregation::AggregationAnalysis;
pub use boundness::BoundnessAnalysis;
pub use demand_attr::DemandAttributeAnalysis;
pub use hidden_relation::HiddenRelationAnalysis;
pub use input_files::InputFilesAnalysis;
pub use invalid_wildcard::InvalidWildcardAnalyzer;
pub use output_files::OutputFilesAnalysis;
pub use type_inference::TypeInference;

pub mod errors {
  pub use super::aggregation::AggregationAnalysisError;
  pub use super::boundness::BoundnessAnalysisError;
  pub use super::demand_attr::DemandAttributeError;
  pub use super::input_files::InputFilesError;
  pub use super::invalid_wildcard::InvalidWildcardError;
  pub use super::output_files::OutputFilesError;
  pub use super::type_inference::TypeInferenceError;
}
