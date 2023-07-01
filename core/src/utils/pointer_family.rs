use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

/// Pointer Family is a trait to generalize reference counted pointers
/// such as `Rc` and `Arc`.
/// Each Pointer Family defines three types: `Pointer<T>` and `Cell<T>`,
/// where `Pointer` is the simple reference counted pointer and `Cell`
/// contains a internally mutable pointer.
pub trait PointerFamily: Clone + PartialEq + 'static {
  /* ==================== Ref Counted ==================== */

  /// Reference counted pointer
  type Rc<T>: Deref<Target = T>;

  /// Create a new `Pointer<T>`
  fn new_rc<T>(value: T) -> Self::Rc<T>;

  /// Clone a `Rc<T>`. Only the reference counter will increase;
  /// the content will not be cloned
  fn clone_rc<T>(ptr: &Self::Rc<T>) -> Self::Rc<T>;

  /// Get an immutable reference to the content pointed by the `Rc`
  fn get_rc<T>(ptr: &Self::Rc<T>) -> &T;

  /// Get a mutable reference to the content pointed by the `Rc`.
  /// Note that the `Rc` itself needs to be mutable here
  fn get_rc_mut<T>(ptr: &mut Self::Rc<T>) -> &mut T;

  /* ==================== Cell ==================== */

  /// Cell
  type Cell<T>;

  /// Create a new `Cell<T>`
  fn new_cell<T>(value: T) -> Self::Cell<T>;

  /// Clone a `Cell<T>`
  fn clone_cell<T: Clone>(ptr: &Self::Cell<T>) -> Self::Cell<T>;

  /// Apply function `f` to the immutable content in the cell
  fn get_cell<T, F, O>(ptr: &Self::Cell<T>, f: F) -> O
  where
    F: FnOnce(&T) -> O;

  /// Apply function `f` to the mutable content in the cell
  fn get_cell_mut<T, F, O>(ptr: &Self::Cell<T>, f: F) -> O
  where
    F: FnOnce(&mut T) -> O;

  /* ==================== Ref Counted Cell ==================== */

  /// Reference counted Cell
  type RcCell<T>;

  /// Create a new `RcCell<T>`
  fn new_rc_cell<T>(value: T) -> Self::RcCell<T>;

  /// Clone a `RcCell<T>`. Only the reference counter will increase;
  /// the content will not be cloned
  fn clone_rc_cell<T>(ptr: &Self::RcCell<T>) -> Self::RcCell<T>;

  /// Clone the internal of the ref counted cell
  fn clone_rc_cell_internal<T: Clone>(ptr: &Self::RcCell<T>) -> T {
    Self::get_rc_cell(ptr, |x| x.clone())
  }

  /// Apply function `f` to the immutable content in the cell
  fn get_rc_cell<T, F, O>(ptr: &Self::RcCell<T>, f: F) -> O
  where
    F: FnOnce(&T) -> O;

  /// Apply function `f` to the mutable content in the cell
  fn get_rc_cell_mut<T, F, O>(ptr: &Self::RcCell<T>, f: F) -> O
  where
    F: FnOnce(&mut T) -> O;
}

/// The Arc pointer family, mainly used for multi-threaded program
#[derive(Clone, Debug, PartialEq)]
pub struct ArcFamily;

impl PointerFamily for ArcFamily {
  type Rc<T> = Arc<T>;

  fn new_rc<T>(value: T) -> Self::Rc<T> {
    Arc::new(value)
  }

  fn clone_rc<T>(ptr: &Self::Rc<T>) -> Self::Rc<T> {
    Arc::clone(ptr)
  }

  fn get_rc<T>(ptr: &Self::Rc<T>) -> &T {
    &*ptr
  }

  fn get_rc_mut<T>(ptr: &mut Self::Rc<T>) -> &mut T {
    Arc::get_mut(ptr).unwrap()
  }

  type Cell<T> = Mutex<T>;

  fn new_cell<T>(value: T) -> Self::Cell<T> {
    Self::Cell::new(value)
  }

  fn clone_cell<T: Clone>(ptr: &Self::Cell<T>) -> Self::Cell<T> {
    Self::Cell::new(ptr.lock().unwrap().clone())
  }

  fn get_cell<T, F, O>(ptr: &Self::Cell<T>, f: F) -> O
  where
    F: FnOnce(&T) -> O,
  {
    f(&ptr.lock().unwrap())
  }

  fn get_cell_mut<T, F, O>(ptr: &Self::Cell<T>, f: F) -> O
  where
    F: FnOnce(&mut T) -> O,
  {
    f(&mut ptr.lock().unwrap())
  }

  type RcCell<T> = Arc<Mutex<T>>;

  fn new_rc_cell<T>(value: T) -> Self::RcCell<T> {
    Arc::new(Mutex::new(value))
  }

  fn clone_rc_cell<T>(ptr: &Self::RcCell<T>) -> Self::RcCell<T> {
    ptr.clone()
  }

  fn get_rc_cell<T, F, O>(ptr: &Self::RcCell<T>, f: F) -> O
  where
    F: FnOnce(&T) -> O,
  {
    f(&*ptr.lock().unwrap())
  }

  fn get_rc_cell_mut<T, F, O>(ptr: &Self::RcCell<T>, f: F) -> O
  where
    F: FnOnce(&mut T) -> O,
  {
    f(&mut (*ptr.lock().unwrap()))
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RcFamily;

impl PointerFamily for RcFamily {
  type Rc<T> = Rc<T>;

  fn new_rc<T>(value: T) -> Self::Rc<T> {
    Rc::new(value)
  }

  fn clone_rc<T>(ptr: &Self::Rc<T>) -> Self::Rc<T> {
    Rc::clone(ptr)
  }

  fn get_rc<T>(ptr: &Self::Rc<T>) -> &T {
    &*ptr
  }

  fn get_rc_mut<T>(ptr: &mut Self::Rc<T>) -> &mut T {
    Rc::get_mut(ptr).unwrap()
  }

  type Cell<T> = RefCell<T>;

  fn new_cell<T>(value: T) -> Self::Cell<T> {
    RefCell::new(value)
  }

  fn clone_cell<T: Clone>(ptr: &Self::Cell<T>) -> Self::Cell<T> {
    RefCell::new(ptr.borrow().clone())
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

  type RcCell<T> = Rc<RefCell<T>>;

  fn new_rc_cell<T>(value: T) -> Self::RcCell<T> {
    Rc::new(RefCell::new(value))
  }

  fn clone_rc_cell<T>(ptr: &Self::RcCell<T>) -> Self::RcCell<T> {
    ptr.clone()
  }

  fn get_rc_cell<T, F, O>(ptr: &Self::RcCell<T>, f: F) -> O
  where
    F: FnOnce(&T) -> O,
  {
    f(&*ptr.borrow())
  }

  fn get_rc_cell_mut<T, F, O>(ptr: &Self::RcCell<T>, f: F) -> O
  where
    F: FnOnce(&mut T) -> O,
  {
    f(&mut (*ptr.borrow_mut()))
  }
}
