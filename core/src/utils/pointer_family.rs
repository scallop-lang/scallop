use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

pub trait PointerFamily: Clone + PartialEq + 'static {
  type Pointer<T>: Deref<Target = T>;

  fn new<T>(value: T) -> Self::Pointer<T>;

  fn clone_ptr<T>(ptr: &Self::Pointer<T>) -> Self::Pointer<T>;

  fn get<T>(ptr: &Self::Pointer<T>) -> &T;

  fn get_mut<T>(ptr: &mut Self::Pointer<T>) -> &mut T;

  type Cell<T>;

  fn new_cell<T>(value: T) -> Self::Cell<T>;

  fn clone_cell<T>(ptr: &Self::Cell<T>) -> Self::Cell<T>;

  fn get_cell<T, F, O>(ptr: &Self::Cell<T>, f: F) -> O
  where
    F: FnOnce(&T) -> O;

  fn get_cell_mut<T, F, O>(ptr: &Self::Cell<T>, f: F) -> O
  where
    F: FnOnce(&mut T) -> O;
}

#[derive(Clone, Debug, PartialEq)]
pub struct ArcFamily;

impl PointerFamily for ArcFamily {
  type Pointer<T> = Arc<T>;

  fn new<T>(value: T) -> Self::Pointer<T> {
    Arc::new(value)
  }

  fn clone_ptr<T>(ptr: &Self::Pointer<T>) -> Self::Pointer<T> {
    Arc::clone(ptr)
  }

  fn get<T>(ptr: &Self::Pointer<T>) -> &T {
    &*ptr
  }

  fn get_mut<T>(ptr: &mut Self::Pointer<T>) -> &mut T {
    Arc::get_mut(ptr).unwrap()
  }

  type Cell<T> = Arc<Mutex<T>>;

  fn new_cell<T>(value: T) -> Self::Cell<T> {
    Arc::new(Mutex::new(value))
  }

  fn clone_cell<T>(ptr: &Self::Cell<T>) -> Self::Cell<T> {
    ptr.clone()
  }

  fn get_cell<T, F, O>(ptr: &Self::Cell<T>, f: F) -> O
  where
    F: FnOnce(&T) -> O,
  {
    f(&*ptr.lock().unwrap())
  }

  fn get_cell_mut<T, F, O>(ptr: &Self::Cell<T>, f: F) -> O
  where
    F: FnOnce(&mut T) -> O,
  {
    f(&mut (*ptr.lock().unwrap()))
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RcFamily;

impl PointerFamily for RcFamily {
  type Pointer<T> = Rc<T>;

  fn new<T>(value: T) -> Self::Pointer<T> {
    Rc::new(value)
  }

  fn clone_ptr<T>(ptr: &Self::Pointer<T>) -> Self::Pointer<T> {
    Rc::clone(ptr)
  }

  fn get<T>(ptr: &Self::Pointer<T>) -> &T {
    &*ptr
  }

  fn get_mut<T>(ptr: &mut Self::Pointer<T>) -> &mut T {
    Rc::get_mut(ptr).unwrap()
  }

  type Cell<T> = Rc<RefCell<T>>;

  fn new_cell<T>(value: T) -> Self::Cell<T> {
    Rc::new(RefCell::new(value))
  }

  fn clone_cell<T>(ptr: &Self::Cell<T>) -> Self::Cell<T> {
    ptr.clone()
  }

  fn get_cell<T, F, O>(ptr: &Self::Cell<T>, f: F) -> O
  where
    F: FnOnce(&T) -> O,
  {
    f(&*ptr.borrow())
  }

  fn get_cell_mut<T, F, O>(ptr: &Self::Cell<T>, f: F) -> O
  where
    F: FnOnce(&mut T) -> O,
  {
    f(&mut (*ptr.borrow_mut()))
  }
}
