use std::sync::*;

use crate::common::foreign_tensor::*;

/// A reference counted
#[derive(Clone)]
pub struct TensorRegistry2 {
  registry: Arc<Mutex<DynamicTensorRegistry>>,
}

impl TensorRegistry2 {
  pub fn new() -> Self {
    Self {
      registry: Arc::new(Mutex::new(DynamicTensorRegistry::new())),
    }
  }

  pub fn set<T: TensorRegistry>(&self, t: T) {
    self.registry.lock().unwrap().set(t)
  }

  pub fn register(&self, tensor: DynamicExternalTensor) -> Option<InternalTensorSymbol> {
    self.registry.lock().unwrap().register(tensor)
  }

  pub fn get(&self, symbol: &InternalTensorSymbol) -> Option<DynamicExternalTensor> {
    self.registry.lock().unwrap().get(symbol)
  }

  pub fn eval(&self, value: &TensorValue) -> Option<DynamicExternalTensor> {
    self.registry.lock().unwrap().eval(value)
  }
}
