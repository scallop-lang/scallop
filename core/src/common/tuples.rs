use super::tuple::*;

pub trait Tuples {
  fn minimum(self) -> Vec<Tuple>;

  fn arg_minimum(self) -> Vec<Tuple>;

  fn maximum(self) -> Vec<Tuple>;

  fn arg_maximum(self) -> Vec<Tuple>;
}

impl<'a, I> Tuples for I
where
  I: Iterator<Item = &'a Tuple>,
{
  fn minimum(self) -> Vec<Tuple> {
    let mut result = vec![];
    let mut min_value = None;
    for v in self {
      if let Some(m) = &min_value {
        if v == m {
          result.push(v.clone());
        } else if v < m {
          min_value = Some(v.clone());
          result.clear();
          result.push(v.clone());
        }
      } else {
        min_value = Some(v.clone());
        result.push(v.clone());
      }
    }
    return result;
  }

  fn arg_minimum(self) -> Vec<Tuple> {
    let mut result = vec![];
    let mut min_value = None;
    for v in self {
      if let Some(m) = &min_value {
        if &v[1] == m {
          result.push(v.clone());
        } else if &v[1] < m {
          min_value = Some(v[1].clone());
          result.clear();
          result.push(v.clone());
        }
      } else {
        min_value = Some(v[1].clone());
        result.push(v.clone());
      }
    }
    return result;
  }

  fn maximum(self) -> Vec<Tuple> {
    let mut result = vec![];
    let mut min_value = None;
    for v in self {
      if let Some(m) = &min_value {
        if v == m {
          result.push(v.clone());
        } else if v > m {
          min_value = Some(v.clone());
          result.clear();
          result.push(v.clone());
        }
      } else {
        min_value = Some(v.clone());
        result.push(v.clone());
      }
    }
    return result;
  }

  fn arg_maximum(self) -> Vec<Tuple> {
    let mut result = vec![];
    let mut min_value = None;
    for v in self {
      if let Some(m) = &min_value {
        if &v[1] == m {
          result.push(v.clone());
        } else if &v[1] > m {
          min_value = Some(v[1].clone());
          result.clear();
          result.push(v.clone());
        }
      } else {
        min_value = Some(v[1].clone());
        result.push(v.clone());
      }
    }
    return result;
  }
}
