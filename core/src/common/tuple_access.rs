pub const TUPLE_ACCESSOR_DEPTH: usize = 3;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TupleAccessor {
  pub len: i8,
  pub indices: [i8; TUPLE_ACCESSOR_DEPTH],
}

impl TupleAccessor {
  pub fn empty() -> Self {
    Self {
      len: 0,
      indices: [0; TUPLE_ACCESSOR_DEPTH],
    }
  }

  pub fn from_index(i: i8) -> Self {
    Self {
      len: 1,
      indices: [i, 0, 0],
    }
  }

  pub fn shift(&self) -> Self {
    if self.len > 0 {
      Self {
        len: self.len - 1,
        indices: [self.indices[1], self.indices[2], 0],
      }
    } else {
      panic!("Cannot shift accessor {:?}", self)
    }
  }

  pub fn prepend(&self, i: i8) -> Self {
    if (self.len as usize) < TUPLE_ACCESSOR_DEPTH {
      Self {
        len: self.len + 1,
        indices: [i, self.indices[0], self.indices[1]],
      }
    } else {
      panic!("Cannot prepend {} to accessor {:?}", i, self)
    }
  }

  pub fn len(&self) -> usize {
    self.len as usize
  }

  pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
    self.indices.iter().take(self.len as usize).map(|i| *i as usize)
  }
}

impl From<usize> for TupleAccessor {
  fn from(u: usize) -> Self {
    Self::from_index(u as i8)
  }
}

impl From<()> for TupleAccessor {
  fn from((): ()) -> Self {
    Self::empty()
  }
}

impl From<(usize, usize)> for TupleAccessor {
  fn from((i1, i2): (usize, usize)) -> Self {
    Self {
      len: 2,
      indices: [i1 as i8, i2 as i8, 0],
    }
  }
}

impl From<(usize, usize, usize)> for TupleAccessor {
  fn from((i1, i2, i3): (usize, usize, usize)) -> Self {
    Self {
      len: 2,
      indices: [i1 as i8, i2 as i8, i3 as i8],
    }
  }
}

impl std::fmt::Debug for TupleAccessor {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_list().entries(&self.indices[0..self.len as usize]).finish()
  }
}
