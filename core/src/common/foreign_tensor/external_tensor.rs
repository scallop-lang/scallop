use serde::*;
use std::any::Any;

use super::*;

pub trait ExternalTensor: 'static + dyn_clone::DynClone + Send + Sync {
  fn shape(&self) -> TensorShape;

  fn get_f64(&self) -> f64;

  fn as_any(&self) -> &dyn Any;
}

pub struct DynamicExternalTensor {
  tensor: Box<dyn ExternalTensor>,
}

impl DynamicExternalTensor {
  pub fn new<T: ExternalTensor>(t: T) -> Self {
    Self { tensor: Box::new(t) }
  }

  pub fn internal(&self) -> &dyn ExternalTensor {
    &*self.tensor
  }

  pub fn cast<T: ExternalTensor>(&self) -> Option<&T> {
    self.internal().as_any().downcast_ref::<T>()
  }
}

impl Clone for DynamicExternalTensor {
  fn clone(&self) -> Self {
    Self {
      tensor: dyn_clone::clone_box(&*self.tensor),
    }
  }
}

/// External tensors are always the same as one another
impl std::cmp::PartialEq for DynamicExternalTensor {
  fn eq(&self, _: &Self) -> bool {
    true
  }
}

impl std::cmp::Eq for DynamicExternalTensor {}

impl std::cmp::PartialOrd for DynamicExternalTensor {
  fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
    Some(std::cmp::Ordering::Equal)
  }
}

impl std::cmp::Ord for DynamicExternalTensor {
  fn cmp(&self, _: &Self) -> std::cmp::Ordering {
    std::cmp::Ordering::Equal
  }
}

impl Serialize for DynamicExternalTensor {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    use serde::ser::*;
    serializer.serialize_struct("DynamicExternalTensor", 0)?.end()
  }
}

impl ExternalTensor for DynamicExternalTensor {
  fn shape(&self) -> TensorShape {
    self.tensor.shape()
  }

  fn get_f64(&self) -> f64 {
    self.tensor.get_f64()
  }

  fn as_any(&self) -> &dyn Any {
    self
  }
}

impl std::fmt::Debug for DynamicExternalTensor {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("DynamicExternalTensor").finish()
  }
}

impl std::fmt::Display for DynamicExternalTensor {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("[tensor]")
  }
}
