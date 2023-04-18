#[derive(Debug, Clone, Default)]
pub struct IdAllocator {
  id: usize,
}

impl IdAllocator {
  pub fn new() -> Self {
    Self { id: 0 }
  }

  pub fn new_with_start(start: usize) -> Self {
    Self { id: start }
  }

  pub fn alloc(&mut self) -> usize {
    let result = self.id;
    self.id += 1;
    result
  }
}
