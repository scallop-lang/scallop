use super::*;

/// Log (base e) foreign function
///
/// ``` scl
/// extern fn $log<T: Float>(x: T) -> T
/// ```
#[derive(Clone)]
pub struct Log;

impl UnaryFloatFunction for Log {
  fn name(&self) -> String {
    "log".to_string()
  }

  fn execute_f32_partial(&self, arg: f32) -> Option<f32> {
    if arg > 0.0 {
      Some(arg.ln())
    } else {
      None
    }
  }

  fn execute_f64_partial(&self, arg: f64) -> Option<f64> {
    if arg > 0.0 {
      Some(arg.ln())
    } else {
      None
    }
  }
}
