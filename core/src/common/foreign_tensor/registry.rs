use super::*;

pub trait TensorRegistry: dyn_clone::DynClone + 'static + Send + Sync {
  fn register(&mut self, ext_tensor: DynamicExternalTensor) -> InternalTensorSymbol;

  fn get(&self, int_tensor: &InternalTensorSymbol) -> Option<&DynamicExternalTensor>;

  fn eval_expr(&self, value: &TensorExpr) -> DynamicExternalTensor;

  fn eval(&self, value: &TensorValue) -> DynamicExternalTensor {
    self.eval_expr(&value.expr)
  }
}

pub struct DynamicTensorRegistry {
  maybe_registry: Option<Box<dyn TensorRegistry>>,
}

impl DynamicTensorRegistry {
  pub fn new() -> Self {
    Self { maybe_registry: None }
  }

  pub fn set<T: TensorRegistry>(&mut self, t: T) {
    self.maybe_registry = Some(Box::new(t));
  }

  pub fn register(&mut self, ext_tensor: DynamicExternalTensor) -> Option<InternalTensorSymbol> {
    if let Some(registry) = &mut self.maybe_registry {
      Some(registry.register(ext_tensor))
    } else {
      None
    }
  }

  pub fn get(&self, int_tensor: &InternalTensorSymbol) -> Option<DynamicExternalTensor> {
    self
      .maybe_registry
      .as_ref()
      .and_then(|registry| registry.get(int_tensor).cloned())
  }

  pub fn eval(&self, value: &TensorValue) -> Option<DynamicExternalTensor> {
    self.maybe_registry.as_ref().map(|registry| registry.eval(value))
  }
}
