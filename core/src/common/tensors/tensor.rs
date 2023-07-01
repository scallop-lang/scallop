use serde::*;

use super::*;

/// An actual tensor containing
pub struct Tensor {
  #[cfg(feature = "torch-tensor")]
  pub tensor: TorchTensor,
}

impl Clone for Tensor {
  #[cfg(feature = "torch-tensor")]
  fn clone(&self) -> Self {
    Self {
      tensor: self.tensor.shallow_clone(),
    }
  }

  #[cfg(not(feature = "torch-tensor"))]
  fn clone(&self) -> Self {
    Self {}
  }
}

impl std::fmt::Debug for Tensor {
  #[cfg(feature = "torch-tensor")]
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("Tensor(<{:?}>)", self.tensor.as_ptr()))
  }

  #[cfg(not(feature = "torch-tensor"))]
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("Tensor")
  }
}

impl std::fmt::Display for Tensor {
  #[cfg(feature = "torch-tensor")]
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!("Tensor({})", self.tensor))
  }

  #[cfg(not(feature = "torch-tensor"))]
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("Tensor")
  }
}

impl std::cmp::PartialEq for Tensor {
  fn eq(&self, _: &Self) -> bool {
    true
  }
}

impl std::cmp::Eq for Tensor {}

impl std::cmp::PartialOrd for Tensor {
  fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
    Some(std::cmp::Ordering::Equal)
  }
}

impl std::cmp::Ord for Tensor {
  fn cmp(&self, _: &Self) -> std::cmp::Ordering {
    std::cmp::Ordering::Equal
  }
}

unsafe impl Send for Tensor {}

unsafe impl Sync for Tensor {}

impl Tensor {
  #[cfg(feature = "torch-tensor")]
  pub fn new(tensor: TorchTensor) -> Self {
    Self { tensor }
  }

  #[cfg(feature = "torch-tensor")]
  pub fn shape(&self) -> TensorShape {
    TensorShape::from(self.tensor.size())
  }

  #[cfg(not(feature = "torch-tensor"))]
  pub fn shape(&self) -> TensorShape {
    TensorShape::scalar()
  }

  #[cfg(feature = "torch-tensor")]
  pub fn get_f64(&self) -> f64 {
    self.tensor.double_value(&[])
  }

  #[cfg(not(feature = "torch-tensor"))]
  pub fn get_f64(&self) -> f64 {
    panic!("{}", NO_TORCH_MSG)
  }
}

impl Serialize for Tensor {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    use serde::ser::*;
    serializer.serialize_struct("Tensor", 0)?.end()
  }
}
