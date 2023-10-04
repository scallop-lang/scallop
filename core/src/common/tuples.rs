use super::tuple::*;

pub trait Tuples<'a> {
  fn sort(self, num_args: usize) -> Vec<&'a Tuple>;

  fn minimum(self) -> Vec<&'a Tuple>;

  fn arg_minimum(self, num_args: usize) -> Vec<&'a Tuple>;

  fn maximum(self) -> Vec<&'a Tuple>;

  fn arg_maximum(self, num_args: usize) -> Vec<&'a Tuple>;
}

impl<'a, I> Tuples<'a> for I
where
  I: Iterator<Item = &'a Tuple>,
{
  fn sort(self, num_args: usize) -> Vec<&'a Tuple> {
    let mut collected = self.collect::<Vec<_>>();
    collected.sort_by_key(|e| &e[num_args..]);
    collected
  }

  fn minimum(self) -> Vec<&'a Tuple> {
    let mut result = vec![];
    let mut min_value = None;
    for v in self {
      if let Some(m) = min_value {
        if v == m {
          result.push(v);
        } else if v < m {
          min_value = Some(v);
          result.clear();
          result.push(v);
        }
      } else {
        min_value = Some(v);
        result.push(v);
      }
    }
    return result;
  }

  fn arg_minimum(self, num_args: usize) -> Vec<&'a Tuple> {
    let mut result = vec![];
    let mut min_value: Option<&[Tuple]> = None;
    for v in self {
      if let Some(m) = &min_value {
        if &v[num_args..] == &m[..] {
          result.push(v);
        } else if &v[num_args..] < m {
          min_value = Some(&v[num_args..]);
          result.clear();
          result.push(v);
        }
      } else {
        min_value = Some(&v[num_args..]);
        result.push(v);
      }
    }
    return result;
  }

  fn maximum(self) -> Vec<&'a Tuple> {
    let mut result = vec![];
    let mut min_value = None;
    for v in self {
      if let Some(m) = min_value {
        if v == m {
          result.push(v);
        } else if v > m {
          min_value = Some(v);
          result.clear();
          result.push(v);
        }
      } else {
        min_value = Some(v);
        result.push(v);
      }
    }
    return result;
  }

  fn arg_maximum(self, num_args: usize) -> Vec<&'a Tuple> {
    let mut result = vec![];
    let mut max_value: Option<&[Tuple]> = None;
    for v in self {
      if let Some(m) = &max_value {
        if &v[num_args..] == &m[..] {
          result.push(v);
        } else if &v[num_args..] > &m[..] {
          max_value = Some(&v[num_args..]);
          result.clear();
          result.push(v);
        }
      } else {
        max_value = Some(&v[num_args..]);
        result.push(v);
      }
    }
    return result;
  }
}
