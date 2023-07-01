use super::*;

/// Range foreign predicate
///
/// ``` scl
/// extern pred range<T: Integer>(begin: T, end: T, i: T)[bbf]
/// ```
///
/// The first two arguments `begin` and `end` are bounded to generate `i` as the free variable.
/// The generated number `i` will be sorted from `begin` (inclusive) to `end` (exclusive).
#[derive(Clone)]
pub struct RangeBBF {
  /// The type of the range operator
  pub ty: ValueType,
}

impl RangeBBF {
  /// Create a new range (bbf) foreign predicate
  pub fn new(ty: ValueType) -> Self {
    Self { ty }
  }

  /// Compute the numbers between
  fn range<T: Integer>(begin: &Value, end: &Value) -> impl Iterator<Item = T>
  where
    Value: TryInto<T>,
  {
    pub struct StepIterator<T: Integer> {
      curr: T,
      end: T,
    }

    impl<T: Integer> Iterator for StepIterator<T> {
      type Item = T;

      fn next(&mut self) -> Option<Self::Item> {
        if self.curr >= self.end {
          None
        } else {
          let result = Some(self.curr);
          self.curr = self.curr + T::one();
          result
        }
      }
    }

    // Cast value into integer type
    let begin: T = begin.clone().try_into().unwrap_or(T::zero());
    let end: T = end.clone().try_into().unwrap_or(T::zero());

    // Finally generate the list
    StepIterator { curr: begin, end }
  }

  fn dyn_range<T: Integer + Into<Value>>(begin: &Value, end: &Value) -> Vec<(DynamicInputTag, Vec<Value>)>
  where
    Value: TryInto<T>,
  {
    Self::range::<T>(begin, end)
      .map(|i| (DynamicInputTag::None, vec![i.into()]))
      .collect()
  }
}

impl ForeignPredicate for RangeBBF {
  fn name(&self) -> String {
    "range".to_string()
  }

  fn generic_type_parameters(&self) -> Vec<ValueType> {
    vec![self.ty.clone()]
  }

  fn arity(&self) -> usize {
    3
  }

  fn argument_type(&self, i: usize) -> ValueType {
    assert!(i < 3);
    self.ty.clone()
  }

  fn num_bounded(&self) -> usize {
    2
  }

  fn evaluate(&self, bounded: &[Value]) -> Vec<(DynamicInputTag, Vec<Value>)> {
    assert_eq!(bounded.len(), 2);
    let begin = &bounded[0];
    let end = &bounded[1];
    match &self.ty {
      ValueType::I8 => Self::dyn_range::<i8>(begin, end),
      ValueType::I16 => Self::dyn_range::<i16>(begin, end),
      ValueType::I32 => Self::dyn_range::<i32>(begin, end),
      ValueType::I64 => Self::dyn_range::<i64>(begin, end),
      ValueType::I128 => Self::dyn_range::<i128>(begin, end),
      ValueType::ISize => Self::dyn_range::<isize>(begin, end),
      ValueType::U8 => Self::dyn_range::<u8>(begin, end),
      ValueType::U16 => Self::dyn_range::<u16>(begin, end),
      ValueType::U32 => Self::dyn_range::<u32>(begin, end),
      ValueType::U64 => Self::dyn_range::<u64>(begin, end),
      ValueType::U128 => Self::dyn_range::<u128>(begin, end),
      ValueType::USize => Self::dyn_range::<usize>(begin, end),
      _ => vec![],
    }
  }
}
