//! The sigmoid functions defined for soft comparison
//!
//! Reference: <https://arxiv.org/pdf/2203.09630.pdf>, Section 4.2

/// Sigmoid (s-shaped) function
#[derive(Clone, Debug)]
pub enum SigmoidFunction {
  Logistic { beta: f64 },
  Reciprocal { beta: f64 },
  Cauchy { beta: f64 },
  OptimalMonotonic { beta: f64 },
}

impl Default for SigmoidFunction {
  fn default() -> Self {
    Self::Logistic { beta: 1.0 }
  }
}

impl SigmoidFunction {
  /// Create a logistic sigmoid function with beta as scaling parameter
  pub fn logistic(beta: f64) -> Self {
    Self::Logistic { beta }
  }

  /// Create a reciprocal sigmoid function with beta as scaling parameter
  pub fn reciprocal(beta: f64) -> Self {
    Self::Reciprocal { beta }
  }

  /// Create a cauchy sigmoid function with beta as scaling parameter
  pub fn cauchy(beta: f64) -> Self {
    Self::Cauchy { beta }
  }

  /// Create a optimal sigmoid function with beta as scaling parameter
  pub fn optimal_monotonic(beta: f64) -> Self {
    Self::OptimalMonotonic { beta }
  }

  /// Evaluate the sigmoid function on input `x`, returns a value in [0, 1]
  pub fn eval(&self, x: f64) -> f64 {
    match self {
      Self::Logistic { beta } => {
        // \frac{1}{1 + e^{- \beta x}}
        1.0 / (1.0 + (-beta * x).exp())
      }
      Self::Reciprocal { beta } => {
        // \frac{\beta x}{1 + 2 \beta |x|} + \frac{1}{2}
        (beta * x) / (1.0 + 2.0 * beta * x.abs()) + 0.5
      }
      Self::Cauchy { beta } => {
        // \frac{1}{\pi} \text{arctan}(\beta x) + \frac{1}{2}
        std::f64::consts::FRAC_1_PI * (beta * x).atan() + 0.5
      }
      Self::OptimalMonotonic { beta } => {
        let xp = beta * x;
        if xp < -0.25 {
          -1.0 / (16.0 * xp)
        } else if xp > 0.25 {
          1.0 - 1.0 / (16.0 * xp)
        } else {
          xp + 0.5
        }
      }
    }
  }

  /// Evaluate the normalized derivative of the sigmoid function
  ///
  /// When x = 0, returns 1;
  /// When x goes to +-infinity, returns 0
  pub fn eval_deriv(&self, x: f64) -> f64 {
    match self {
      Self::Logistic { beta } => {
        // sech^2(\frac{\beta x}{2})
        let v = 0.5 * beta * x;
        let epv = std::f64::consts::E.powf(v);
        let epv2 = epv.powi(2);
        2.0 * epv / (epv2 + 1.0)
      }
      Self::Reciprocal { .. } => {
        unimplemented!()
      }
      Self::Cauchy { .. } => {
        unimplemented!()
      }
      Self::OptimalMonotonic { .. } => {
        unimplemented!()
      }
    }
  }
}
