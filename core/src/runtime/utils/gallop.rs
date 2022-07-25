// pub(crate) fn gallop<T>(mut slice: &[T], mut cmp: impl FnMut(&T) -> bool) -> &[T] {
//   // if empty slice, or already >= element, return
//   if !slice.is_empty() && cmp(&slice[0]) {
//     let mut step = 1;
//     while step < slice.len() && cmp(&slice[step]) {
//       slice = &slice[step..];
//       step <<= 1;
//     }

//     step >>= 1;
//     while step > 0 {
//       if step < slice.len() && cmp(&slice[step]) {
//         slice = &slice[step..];
//       }
//       step >>= 1;
//     }

//     slice = &slice[1..]; // advance one, as we always stayed < value
//   }

//   slice
// }

pub(crate) fn gallop_index<T>(
  slice: &[T],
  mut begin: usize,
  mut cmp: impl FnMut(&T) -> bool,
) -> usize {
  // if empty slice, or already >= element, return
  if !slice.is_empty() && cmp(&slice[0]) {
    let mut step = 1;
    while begin + step < slice.len() && cmp(&slice[begin + step]) {
      begin += step;
      step <<= 1;
    }

    step >>= 1;
    while step > 0 {
      if begin + step < slice.len() && cmp(&slice[begin + step]) {
        begin += step;
      }
      step >>= 1;
    }

    begin += 1; // advance one, as we always stayed < value
  }

  begin
}
