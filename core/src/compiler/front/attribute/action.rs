use crate::compiler::front::*;

/// An action to perform after analyzing an attribute
pub enum AttributeAction {
  /// Remove the current item
  RemoveItem,

  /// Adding a new item
  AddItem(Item),

  /// Replace the current item with a new item
  ReplaceItem(Item),

  /// Multiple actions
  Multiple(Vec<AttributeAction>),

  /// Context process
  Context(Box<dyn FnOnce(&mut FrontContext)>),

  /// Error with a message
  Error(String),

  /// Do nothing
  Nothing,
}
