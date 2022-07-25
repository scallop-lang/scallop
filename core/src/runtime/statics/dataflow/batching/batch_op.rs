pub trait BatchUnaryOp<I1>: Clone {
  type I2;

  fn apply(&self, i1: I1) -> Self::I2;
}

pub trait BatchBinaryOp<I1, I2>: Clone {
  type IOut;

  fn apply(&self, i1: I1, i2: I2) -> Self::IOut;
}
