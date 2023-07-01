use super::*;

/// Arccosine foreign function
///
/// ``` scl
/// extern fn $acos<T: Float>(x: T) -> T
/// ```
#[derive(Clone)]
pub struct Acos;

impl UnaryFloatFunction for Acos {
  fn name(&self) -> String {
    "acos".to_string()
  }

  fn execute_f32_partial(&self, arg: f32) -> Option<f32> {
    if arg >= -1.0 && arg <= 1.0 {
      Some(arg.acos())
    } else {
      None
    }
  }

  fn execute_f64_partial(&self, arg: f64) -> Option<f64> {
    if arg >= -1.0 && arg <= 1.0 {
      Some(arg.acos())
    } else {
      None
    }
  }
}
