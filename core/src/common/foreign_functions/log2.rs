use super::*;

/// Log (base 2) foreign function
///
/// ``` scl
/// extern fn $log2<T: Float>(x: T) -> T
/// ```
#[derive(Clone)]
pub struct Log2;

impl UnaryFloatFunction for Log2 {
  fn name(&self) -> String {
    "log2".to_string()
  }

  fn execute_f32_partial(&self, arg: f32) -> Option<f32> {
    if arg > 0.0 {
      Some(arg.log2())
    } else {
      None
    }
  }

  fn execute_f64_partial(&self, arg: f64) -> Option<f64> {
    if arg > 0.0 {
      Some(arg.log2())
    } else {
      None
    }
  }
}
