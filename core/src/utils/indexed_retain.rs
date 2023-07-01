pub trait IndexedRetain<T> {
  fn retain_with_index<F>(&mut self, f: F)
  where
    F: FnMut(usize, &T) -> bool;
}

impl<T> IndexedRetain<T> for Vec<T> {
  fn retain_with_index<F>(&mut self, mut f: F)
  where
    F: FnMut(usize, &T) -> bool, // the signature of the callback changes
  {
    let len = self.len();
    let mut del = 0;
    {
      let v = &mut **self;

      for i in 0..len {
        // only implementation change here
        if !f(i, &v[i]) {
          del += 1;
        } else if del > 0 {
          v.swap(i - del, i);
        }
      }
    }
    if del > 0 {
      self.truncate(len - del);
    }
  }
}
