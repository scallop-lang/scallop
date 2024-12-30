use crate::runtime::database::*;

use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct StorageAttribute {
  pub metadata: StorageMetadata,
}

impl StorageAttribute {
  pub fn new(metadata: StorageMetadata) -> Self {
    Self {
      metadata,
    }
  }
}

impl AttributeTrait for StorageAttribute {
  fn name(&self) -> String {
    "storage".to_string()
  }

  fn args(&self) -> Vec<String> {
    vec![format!("{:?}", self.metadata.to_string())]
  }
}
