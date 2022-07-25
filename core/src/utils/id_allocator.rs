#[derive(Debug, Clone, Default)]
pub struct IdAllocator {
  id: usize,
}

impl IdAllocator {
  pub fn alloc(&mut self) -> usize {
    let result = self.id;
    self.id += 1;
    result
  }
}
