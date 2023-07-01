use std::collections::*;

use crate::utils::*;

use super::super::*;
use super::*;

pub enum PostProcessingAction {
  RemoveItem { item_index: usize },
  ReplaceItem { item_index: usize, item: Item },
  AddItem { item: Item },
  Context { func: Box<dyn FnOnce(&mut FrontContext)> },
  Error { msg: String },
}

pub struct PostProcessingContext {
  actions: Vec<PostProcessingAction>,
}

impl PostProcessingContext {
  pub fn new() -> Self {
    Self { actions: Vec::new() }
  }

  pub fn add_action(&mut self, action: AttributeAction, item_index: usize) {
    // Helper function
    fn add_action(post_proc_actions: &mut Vec<PostProcessingAction>, action: AttributeAction, item_index: usize) {
      match action {
        AttributeAction::AddItem(item) => {
          post_proc_actions.push(PostProcessingAction::AddItem { item });
        }
        AttributeAction::Context(func) => {
          post_proc_actions.push(PostProcessingAction::Context { func });
        }
        AttributeAction::Error(msg) => {
          post_proc_actions.push(PostProcessingAction::Error { msg });
        }
        AttributeAction::Multiple(acts) => {
          for act in acts {
            add_action(post_proc_actions, act, item_index);
          }
        }
        AttributeAction::Nothing => {}
        AttributeAction::RemoveItem => {
          post_proc_actions.push(PostProcessingAction::RemoveItem { item_index });
        }
        AttributeAction::ReplaceItem(item) => {
          post_proc_actions.push(PostProcessingAction::ReplaceItem { item, item_index });
        }
      }
    }

    // Invoke the helper function
    add_action(&mut self.actions, action, item_index);
  }

  pub fn process(self, ctx: &mut FrontContext, items: &mut Vec<Item>) -> Result<(), AttributeError> {
    let mut to_remove_item_index = HashSet::new();
    let mut new_items = Vec::new();

    for action in self.actions {
      match action {
        PostProcessingAction::AddItem { item } => {
          new_items.push(item);
        }
        PostProcessingAction::Context { func } => {
          func(ctx);
        }
        PostProcessingAction::RemoveItem { item_index } => {
          to_remove_item_index.insert(item_index);
        }
        PostProcessingAction::ReplaceItem { item_index, item } => {
          items[item_index] = item;
        }
        PostProcessingAction::Error { msg } => return Err(AttributeError::Custom { msg }),
      }
    }

    items.retain_with_index(|id, _| !to_remove_item_index.contains(&id));
    items.extend(new_items.into_iter());

    Ok(())
  }
}
