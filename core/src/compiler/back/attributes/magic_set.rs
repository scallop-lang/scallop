use super::*;

/// Magic-Set attributes to the helper relations storing the demanded-tuples for on-demand relations.
/// These relations are called "magic-sets".
#[derive(Clone, Debug, PartialEq)]
pub struct MagicSetAttribute;

impl AttributeTrait for MagicSetAttribute {
  fn name(&self) -> String {
    "magic_set".to_string()
  }
}
