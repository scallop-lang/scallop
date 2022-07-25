mod constant_folding;
mod constant_propagation;
mod demand_transform;
mod empty_rule_to_fact;
mod equality_propagation;
mod remove_false_rules;
mod remove_true_literals;

pub use constant_folding::*;
pub use constant_propagation::*;
pub use demand_transform::*;
pub use empty_rule_to_fact::*;
pub use equality_propagation::*;
pub use remove_false_rules::*;
pub use remove_true_literals::*;
