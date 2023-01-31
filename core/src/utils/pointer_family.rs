use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

/// Pointer Family is a trait to generalize reference counted pointers
/// such as `Rc` and `Arc`.
/// Each Pointer Family defines two types: `Pointer<T>` and `Cell<T>`,
/// where `Pointer` is the simple reference counted pointer and `Cell`
/// contains a internally mutable pointer.
pub trait PointerFamily: Clone + PartialEq + 'static {
  /// Reference counted pointer
  type Pointer<T>: Deref<Target = T>;

  /// Create a new `Pointer<T>`
  fn new<T>(value: T) -> Self::Pointer<T>;

  /// Clone a `Pointer<T>`. Only the reference counter will increase;
  /// the content will not be cloned
  fn clone_ptr<T>(ptr: &Self::Pointer<T>) -> Self::Pointer<T>;

  /// Get an immutable reference to the content pointed by the `Pointer`
  fn get<T>(ptr: &Self::Pointer<T>) -> &T;

  /// Get a mutable reference to the content pointed by the `Pointer`.
  /// Note that the `Pointer` itself needs to be mutable here
  fn get_mut<T>(ptr: &mut Self::Pointer<T>) -> &mut T;

  /// Reference counted Cell
  type Cell<T>;

  /// Create a new `Cell<T>`
  fn new_cell<T>(value: T) -> Self::Cell<T>;

  /// Clone a `Cell<T>`. Only the reference counter will increase;
  /// the content will not be cloned
  fn clone_cell<T>(ptr: &Self::Cell<T>) -> Self::Cell<T>;

  /// Apply function `f` to the immutable content in the cell
  fn get_cell<T, F, O>(ptr: &Self::Cell<T>, f: F) -> O
  where
    F: FnOnce(&T) -> O;

  /// Apply function `f` to the mutable content in the cell
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
