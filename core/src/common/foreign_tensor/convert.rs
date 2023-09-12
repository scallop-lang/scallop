use super::DynamicExternalTensor;

/// The trait defining the piece of information can be converted from Tensor
///
/// For Python, we want the external tag to be accessible from tensor
pub trait FromTensor: Clone + 'static {
  fn from_tensor(tensor: DynamicExternalTensor) -> Option<Self>;
}

impl FromTensor for () {
  #[allow(unused)]
  fn from_tensor(tensor: DynamicExternalTensor) -> Option<Self> {
    None
  }
}
