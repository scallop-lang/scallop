// use std::rc::Rc;

use super::generic_tuple::GenericTuple;
use super::tuple_type::TupleType;
use super::value::Value;

pub type Tuple = GenericTuple<Value>;

impl Tuple {
  pub fn from_primitives(prims: Vec<Value>) -> Self {
    Self::Tuple(prims.into_iter().map(Self::Value).collect())
  }

  pub fn tuple_type(&self) -> TupleType {
    TupleType::type_of(self)
  }

  pub fn as_ref_values(&self) -> Vec<&Value> {
    match self {
      Self::Value(p) => vec![p],
      Self::Tuple(t) => t
        .iter()
        .map(|t| match t {
          Self::Value(v) => v,
          _ => panic!("Not a value"),
        })
        .collect(),
    }
  }

  pub fn as_value(&self) -> Value {
    match self {
      Self::Value(p) => p.clone(),
      _ => panic!("Not a value"),
    }
  }

  pub fn as_i8(&self) -> i8 {
    AsTuple::<i8>::as_tuple(self)
  }

  pub fn as_i16(&self) -> i16 {
    AsTuple::<i16>::as_tuple(self)
  }

  pub fn as_i32(&self) -> i32 {
    AsTuple::<i32>::as_tuple(self)
  }

  pub fn as_i64(&self) -> i64 {
    AsTuple::<i64>::as_tuple(self)
  }

  pub fn as_i128(&self) -> i128 {
    AsTuple::<i128>::as_tuple(self)
  }

  pub fn as_isize(&self) -> isize {
    AsTuple::<isize>::as_tuple(self)
  }

  pub fn as_u8(&self) -> u8 {
    AsTuple::<u8>::as_tuple(self)
  }

  pub fn as_u16(&self) -> u16 {
    AsTuple::<u16>::as_tuple(self)
  }

  pub fn as_u32(&self) -> u32 {
    AsTuple::<u32>::as_tuple(self)
  }

  pub fn as_u64(&self) -> u64 {
    AsTuple::<u64>::as_tuple(self)
  }

  pub fn as_u128(&self) -> u128 {
    AsTuple::<u128>::as_tuple(self)
  }

  pub fn as_usize(&self) -> usize {
    AsTuple::<usize>::as_tuple(self)
  }

  pub fn as_f32(&self) -> f32 {
    AsTuple::<f32>::as_tuple(self)
  }

  pub fn as_f64(&self) -> f64 {
    AsTuple::<f64>::as_tuple(self)
  }

  pub fn as_char(&self) -> char {
    AsTuple::<char>::as_tuple(self)
  }

  pub fn as_bool(&self) -> bool {
    AsTuple::<bool>::as_tuple(self)
  }

  pub fn as_str(&self) -> &'static str {
    AsTuple::<&'static str>::as_tuple(self)
  }

  pub fn as_string(&self) -> String {
    AsTuple::<String>::as_tuple(self)
  }

  // pub fn as_rc_string(&self) -> Rc<String> {
  //   AsTuple::<Rc<String>>::as_tuple(self)
  // }
}

impl std::fmt::Debug for Tuple {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Tuple(tuple) => {
        if tuple.is_empty() {
          f.write_str("()")
        } else {
          let mut t = &mut f.debug_tuple("");
          for elem in tuple.iter() {
            t = t.field(&elem);
          }
          t.finish()
        }
      }
      Self::Value(p) => f.write_fmt(format_args!("{:?}", p)),
    }
  }
}

impl std::fmt::Display for Tuple {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Tuple(tuple) => {
        if tuple.is_empty() {
          f.write_str("()")
        } else {
          f.write_str("(")?;
          for (i, elem) in tuple.iter().enumerate() {
            if i > 0 {
              f.write_str(", ")?;
            }
            std::fmt::Display::fmt(elem, f)?;
          }
          f.write_str(")")
        }
      }
      Self::Value(p) => f.write_fmt(format_args!("{}", p)),
    }
  }
}

impl<A> From<A> for Tuple
where
  A: Into<Value>,
{
  fn from(a: A) -> Self {
    Self::Value(a.into())
  }
}

pub trait AsTuple<T> {
  fn as_tuple(&self) -> T;
}

impl AsTuple<i8> for Tuple {
  fn as_tuple(&self) -> i8 {
    match self {
      Self::Value(Value::I8(i)) => *i,
      _ => panic!("Cannot perform as_tuple<i8>"),
    }
  }
}

impl AsTuple<i16> for Tuple {
  fn as_tuple(&self) -> i16 {
    match self {
      Self::Value(Value::I16(i)) => *i,
      _ => panic!("Cannot perform as_tuple<i16>"),
    }
  }
}

impl AsTuple<i32> for Tuple {
  fn as_tuple(&self) -> i32 {
    match self {
      Self::Value(Value::I32(i)) => *i,
      _ => panic!("Cannot perform as_tuple<i32>"),
    }
  }
}

impl AsTuple<i64> for Tuple {
  fn as_tuple(&self) -> i64 {
    match self {
      Self::Value(Value::I64(i)) => *i,
      _ => panic!("Cannot perform as_tuple<i64>"),
    }
  }
}

impl AsTuple<i128> for Tuple {
  fn as_tuple(&self) -> i128 {
    match self {
      Self::Value(Value::I128(i)) => *i,
      _ => panic!("Cannot perform as_tuple<i128>"),
    }
  }
}

impl AsTuple<isize> for Tuple {
  fn as_tuple(&self) -> isize {
    match self {
      Self::Value(Value::ISize(i)) => *i,
      _ => panic!("Cannot perform as_tuple<isize>"),
    }
  }
}

impl AsTuple<u8> for Tuple {
  fn as_tuple(&self) -> u8 {
    match self {
      Self::Value(Value::U8(i)) => *i,
      _ => panic!("Cannot perform as_tuple<u8>"),
    }
  }
}

impl AsTuple<u16> for Tuple {
  fn as_tuple(&self) -> u16 {
    match self {
      Self::Value(Value::U16(i)) => *i,
      _ => panic!("Cannot perform as_tuple<u16>"),
    }
  }
}

impl AsTuple<u32> for Tuple {
  fn as_tuple(&self) -> u32 {
    match self {
      Self::Value(Value::U32(i)) => *i,
      _ => panic!("Cannot perform as_tuple<u32>"),
    }
  }
}

impl AsTuple<u64> for Tuple {
  fn as_tuple(&self) -> u64 {
    match self {
      Self::Value(Value::U64(i)) => *i,
      _ => panic!("Cannot perform as_tuple<u64>"),
    }
  }
}

impl AsTuple<u128> for Tuple {
  fn as_tuple(&self) -> u128 {
    match self {
      Self::Value(Value::U128(i)) => *i,
      _ => panic!("Cannot perform as_tuple<u128>"),
    }
  }
}

impl AsTuple<usize> for Tuple {
  fn as_tuple(&self) -> usize {
    match self {
      Self::Value(Value::USize(i)) => *i,
      _ => panic!("Cannot perform as_tuple<usize>"),
    }
  }
}

impl AsTuple<f32> for Tuple {
  fn as_tuple(&self) -> f32 {
    match self {
      Self::Value(Value::F32(i)) => *i,
      _ => panic!("Cannot perform as_tuple<f32>"),
    }
  }
}

impl AsTuple<f64> for Tuple {
  fn as_tuple(&self) -> f64 {
    match self {
      Self::Value(Value::F64(i)) => *i,
      _ => panic!("Cannot perform as_tuple<f64>"),
    }
  }
}

impl AsTuple<char> for Tuple {
  fn as_tuple(&self) -> char {
    match self {
      Self::Value(Value::Char(c)) => *c,
      _ => panic!("Cannot perform as_tuple<char>"),
    }
  }
}

impl AsTuple<bool> for Tuple {
  fn as_tuple(&self) -> bool {
    match self {
      Self::Value(Value::Bool(b)) => *b,
      _ => panic!("Cannot perform as_tuple<bool>"),
    }
  }
}

impl AsTuple<&'static str> for Tuple {
  fn as_tuple(&self) -> &'static str {
    match self {
      Self::Value(Value::Str(s)) => s,
      _ => panic!("Cannot perform as_tuple<&str>"),
    }
  }
}

impl AsTuple<String> for Tuple {
  fn as_tuple(&self) -> String {
    match self {
      Self::Value(Value::String(s)) => s.clone(),
      _ => panic!("Cannot perform as_tuple<String>"),
    }
  }
}

// impl AsTuple<Rc<String>> for Tuple {
//   fn as_tuple(&self) -> Rc<String> {
//     match self {
//       Self::Value(Value::RcString(s)) => s.clone(),
//       _ => panic!("Cannot perform as_tuple<Rc<String>>"),
//     }
//   }
// }

impl AsTuple<()> for Tuple {
  fn as_tuple(&self) -> () {
    match self {
      Self::Tuple(v) if v.is_empty() => (),
      _ => panic!("Cannot perform as_tuple<()>"),
    }
  }
}

impl<A> AsTuple<(A,)> for Tuple
where
  Tuple: AsTuple<A>,
{
  fn as_tuple(&self) -> (A,) {
    match self {
      Self::Tuple(l) if l.len() == 1 => (l[0].as_tuple(),),
      _ => panic!("Cannot perform as_tuple<(A,)>"),
    }
  }
}

impl<A, B> AsTuple<(A, B)> for Tuple
where
  Tuple: AsTuple<A>,
  Tuple: AsTuple<B>,
{
  fn as_tuple(&self) -> (A, B) {
    match self {
      Self::Tuple(l) if l.len() == 2 => (l[0].as_tuple(), l[1].as_tuple()),
      _ => panic!("Cannot perform as_tuple<(A, B)>"),
    }
  }
}

impl<A, B, C> AsTuple<(A, B, C)> for Tuple
where
  Tuple: AsTuple<A>,
  Tuple: AsTuple<B>,
  Tuple: AsTuple<C>,
{
  fn as_tuple(&self) -> (A, B, C) {
    match self {
      Self::Tuple(l) if l.len() == 3 => (l[0].as_tuple(), l[1].as_tuple(), l[2].as_tuple()),
      _ => panic!("Cannot perform as_tuple<(A, B, C)>"),
    }
  }
}

impl<A, B, C, D> AsTuple<(A, B, C, D)> for Tuple
where
  Tuple: AsTuple<A>,
  Tuple: AsTuple<B>,
  Tuple: AsTuple<C>,
  Tuple: AsTuple<D>,
{
  fn as_tuple(&self) -> (A, B, C, D) {
    match self {
      Self::Tuple(l) if l.len() == 4 => (
        l[0].as_tuple(),
        l[1].as_tuple(),
        l[2].as_tuple(),
        l[3].as_tuple(),
      ),
      _ => panic!("Cannot perform as_tuple<(A, B, C, D)>"),
    }
  }
}

impl<A, B, C, D, E> AsTuple<(A, B, C, D, E)> for Tuple
where
  Tuple: AsTuple<A>,
  Tuple: AsTuple<B>,
  Tuple: AsTuple<C>,
  Tuple: AsTuple<D>,
  Tuple: AsTuple<E>,
{
  fn as_tuple(&self) -> (A, B, C, D, E) {
    match self {
      Self::Tuple(l) if l.len() == 5 => (
        l[0].as_tuple(),
        l[1].as_tuple(),
        l[2].as_tuple(),
        l[3].as_tuple(),
        l[4].as_tuple(),
      ),
      _ => panic!("Cannot perform as_tuple<(A, B, C, D, E)>"),
    }
  }
}

impl<A, B, C, D, E, F> AsTuple<(A, B, C, D, E, F)> for Tuple
where
  Tuple: AsTuple<A>,
  Tuple: AsTuple<B>,
  Tuple: AsTuple<C>,
  Tuple: AsTuple<D>,
  Tuple: AsTuple<E>,
  Tuple: AsTuple<F>,
{
  fn as_tuple(&self) -> (A, B, C, D, E, F) {
    match self {
      Self::Tuple(l) if l.len() == 6 => (
        l[0].as_tuple(),
        l[1].as_tuple(),
        l[2].as_tuple(),
        l[3].as_tuple(),
        l[4].as_tuple(),
        l[5].as_tuple(),
      ),
      _ => panic!("Cannot perform as_tuple<(A, B, C, D, E, F)>"),
    }
  }
}

impl<A, B, C, D, E, F, G> AsTuple<(A, B, C, D, E, F, G)> for Tuple
where
  Tuple: AsTuple<A>,
  Tuple: AsTuple<B>,
  Tuple: AsTuple<C>,
  Tuple: AsTuple<D>,
  Tuple: AsTuple<E>,
  Tuple: AsTuple<F>,
  Tuple: AsTuple<G>,
{
  fn as_tuple(&self) -> (A, B, C, D, E, F, G) {
    match self {
      Self::Tuple(l) if l.len() == 7 => (
        l[0].as_tuple(),
        l[1].as_tuple(),
        l[2].as_tuple(),
        l[3].as_tuple(),
        l[4].as_tuple(),
        l[5].as_tuple(),
        l[6].as_tuple(),
      ),
      _ => panic!("Cannot perform as_tuple<(A, B, C, D, E, F, G)>"),
    }
  }
}

impl<A, B, C, D, E, F, G, H> AsTuple<(A, B, C, D, E, F, G, H)> for Tuple
where
  Tuple: AsTuple<A>,
  Tuple: AsTuple<B>,
  Tuple: AsTuple<C>,
  Tuple: AsTuple<D>,
  Tuple: AsTuple<E>,
  Tuple: AsTuple<F>,
  Tuple: AsTuple<G>,
  Tuple: AsTuple<H>,
{
  fn as_tuple(&self) -> (A, B, C, D, E, F, G, H) {
    match self {
      Self::Tuple(l) if l.len() == 8 => (
        l[0].as_tuple(),
        l[1].as_tuple(),
        l[2].as_tuple(),
        l[3].as_tuple(),
        l[4].as_tuple(),
        l[5].as_tuple(),
        l[6].as_tuple(),
        l[7].as_tuple(),
      ),
      _ => panic!("Cannot perform as_tuple<(A, B, C, D, E, F, G, H)>"),
    }
  }
}

impl<A, B, C, D, E, F, G, H, I> AsTuple<(A, B, C, D, E, F, G, H, I)> for Tuple
where
  Tuple: AsTuple<A>,
  Tuple: AsTuple<B>,
  Tuple: AsTuple<C>,
  Tuple: AsTuple<D>,
  Tuple: AsTuple<E>,
  Tuple: AsTuple<F>,
  Tuple: AsTuple<G>,
  Tuple: AsTuple<H>,
  Tuple: AsTuple<I>,
{
  fn as_tuple(&self) -> (A, B, C, D, E, F, G, H, I) {
    match self {
      Self::Tuple(l) if l.len() == 9 => (
        l[0].as_tuple(),
        l[1].as_tuple(),
        l[2].as_tuple(),
        l[3].as_tuple(),
        l[4].as_tuple(),
        l[5].as_tuple(),
        l[6].as_tuple(),
        l[7].as_tuple(),
        l[8].as_tuple(),
      ),
      _ => panic!("Cannot perform as_tuple<(A, B, C, D, E, F, G, H, I)>"),
    }
  }
}

impl<A, B, C, D, E, F, G, H, I, J> AsTuple<(A, B, C, D, E, F, G, H, I, J)> for Tuple
where
  Tuple: AsTuple<A>,
  Tuple: AsTuple<B>,
  Tuple: AsTuple<C>,
  Tuple: AsTuple<D>,
  Tuple: AsTuple<E>,
  Tuple: AsTuple<F>,
  Tuple: AsTuple<G>,
  Tuple: AsTuple<H>,
  Tuple: AsTuple<I>,
  Tuple: AsTuple<J>,
{
  fn as_tuple(&self) -> (A, B, C, D, E, F, G, H, I, J) {
    match self {
      Self::Tuple(l) if l.len() == 10 => (
        l[0].as_tuple(),
        l[1].as_tuple(),
        l[2].as_tuple(),
        l[3].as_tuple(),
        l[4].as_tuple(),
        l[5].as_tuple(),
        l[6].as_tuple(),
        l[7].as_tuple(),
        l[8].as_tuple(),
        l[9].as_tuple(),
      ),
      _ => panic!("Cannot perform as_tuple<(A, B, C, D, E, F, G, H, I, J)>"),
    }
  }
}
