use std::sync::*;

use crate::common::tensors::*;

/// A reference counted
#[derive(Clone)]
pub struct TensorRegistry2 {
  registry: Arc<Mutex<TensorRegistry>>,
}

impl TensorRegistry2 {
  pub fn new() -> Self {
    Self {
      registry: Arc::new(Mutex::new(TensorRegistry::new())),
    }
  }

  pub fn register(&self, tensor: Tensor) -> TensorSymbol {
    self.registry.lock().unwrap().register(tensor)
  }

  pub fn get(&self, symbol: &TensorSymbol) -> Option<Tensor> {
    self.registry.lock().unwrap().get(symbol).cloned()
  }

  pub fn eval(&self, value: &TensorValue) -> Tensor {
    self.registry.lock().unwrap().eval(value)
  }
}
