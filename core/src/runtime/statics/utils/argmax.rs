use crate::runtime::statics::*;

pub fn static_argmax<I, T1, T2>(batch: I) -> Vec<(T1, T2)>
where
  T1: StaticTupleTrait,
  T2: StaticTupleTrait,
  I: Iterator<Item = (T1, T2)>,
{
  let mut result = vec![];
  let mut min_value = None;
  for v in batch {
    if let Some(m) = &min_value {
      if &v.1 == m {
        result.push(v.clone());
      } else if &v.1 > m {
        min_value = Some(v.1.clone());
        result.clear();
        result.push(v.clone());
      }
    } else {
      min_value = Some(v.1.clone());
      result.push(v.clone());
    }
  }
  return result;
}
