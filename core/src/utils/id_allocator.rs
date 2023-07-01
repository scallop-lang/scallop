use std::sync::*;

/// An ID allocator
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

/// An alternative ID allocator which has internal mutability
#[derive(Debug, Clone, Default)]
pub struct IdAllocator2 {
  pub id_allocator: Arc<Mutex<IdAllocator>>,
}

impl IdAllocator2 {
  pub fn new() -> Self {
    Self {
      id_allocator: Arc::new(Mutex::new(IdAllocator::new())),
    }
  }

  pub fn new_with_start(start: usize) -> Self {
    Self {
      id_allocator: Arc::new(Mutex::new(IdAllocator::new_with_start(start))),
    }
  }

  pub fn alloc(&self) -> usize {
    self.id_allocator.lock().unwrap().alloc()
  }
}
