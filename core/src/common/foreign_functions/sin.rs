use super::*;

/// Sin value foreign function
///
/// ``` scl
/// extern fn $sin<T: Float>(x: T) -> T
/// ```
#[derive(Clone)]
pub struct Sin;

impl UnaryFloatFunction for Sin {
  fn name(&self) -> String {
    "sin".to_string()
  }

  fn execute_f32(&self, arg: f32) -> f32 {
    arg.sin()
  }

  fn execute_f64(&self, arg: f64) -> f64 {
    arg.sin()
  }
}
