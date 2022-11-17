#[derive(Clone, Debug, Default)]
pub struct CompileOptions {
  // Debug options
  pub debug: bool,
  pub debug_front: bool,
  pub debug_back: bool,
  pub debug_ram: bool,

  // Report front errors
  pub report_front_errors: bool,
  pub report_front_warnings: bool,

  // Back compile options
  pub do_not_remove_unused_relations: bool,
  pub do_not_demand_transform: bool,
  pub do_not_optimize_ram: bool,

  /// Whether static relation type needs to be convertable to dynamic tuple
  pub static_relation_type_as_tuple: bool,

  /// Output including hidden things
  pub output_all: bool,

  // Allow probability
  pub allow_probability: bool,
}
