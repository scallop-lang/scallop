pub mod aggregation;
pub mod algebraic_data_type;
pub mod boundness;
pub mod character_literal;
pub mod constant_decl;
pub mod demand_attr;
pub mod goal_relation;
pub mod head_relation;
pub mod hidden_relation;
pub mod input_files;
pub mod invalid_constant;
pub mod invalid_wildcard;
pub mod output_files;
pub mod scheduler_attr;
pub mod tagged_rule;
pub mod type_inference;

pub use aggregation::AggregationAnalysis;
pub use algebraic_data_type::AlgebraicDataTypeAnalysis;
pub use boundness::BoundnessAnalysis;
pub use character_literal::CharacterLiteralAnalysis;
pub use constant_decl::ConstantDeclAnalysis;
pub use demand_attr::DemandAttributeAnalysis;
pub use goal_relation::GoalRelationAnalysis;
pub use head_relation::HeadRelationAnalysis;
pub use hidden_relation::HiddenRelationAnalysis;
pub use input_files::InputFilesAnalysis;
pub use invalid_constant::InvalidConstantAnalyzer;
pub use invalid_wildcard::InvalidWildcardAnalyzer;
pub use output_files::OutputFilesAnalysis;
pub use scheduler_attr::SchedulerAttributeAnalysis;
pub use tagged_rule::TaggedRuleAnalysis;
pub use type_inference::TypeInference;

pub mod errors {
  pub use super::aggregation::AggregationAnalysisError;
  pub use super::algebraic_data_type::ADTError;
  pub use super::boundness::BoundnessAnalysisError;
  pub use super::constant_decl::ConstantDeclError;
  pub use super::demand_attr::DemandAttributeError;
  pub use super::head_relation::HeadRelationError;
  pub use super::input_files::InputFilesError;
  pub use super::invalid_constant::InvalidConstantError;
  pub use super::invalid_wildcard::InvalidWildcardError;
  pub use super::output_files::OutputFilesError;
}
