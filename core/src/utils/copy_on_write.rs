use super::{PointerFamily, RcFamily};

pub struct CopyOnWrite<T: Clone, P: PointerFamily = RcFamily>(P::Pointer<T>);

impl<T: Clone, P: PointerFamily> CopyOnWrite<T, P> {
  pub fn new(t: T) -> Self {
    Self(P::new(t))
  }

  pub fn borrow(&self) -> &T {
    &*self.0
  }

  pub fn modify<F>(&mut self, f: F)
  where
    F: FnOnce(&mut T),
  {
    let mut new_inner = (*self.0).clone();
    f(&mut new_inner);
    *self = Self(P::new(new_inner));
  }
}

impl<T: Clone, P: PointerFamily> Clone for CopyOnWrite<T, P> {
  fn clone(&self) -> Self {
    Self(P::clone_ptr(&self.0))
  }
}

impl<T: Clone + std::fmt::Debug, P: PointerFamily> std::fmt::Debug for CopyOnWrite<T, P> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    std::fmt::Debug::fmt(&self.borrow(), f)
  }
}
