pub fn collect_line_offset_length(s: &str) -> Vec<(usize, usize)> {
  let line_offsets = std::iter::once(0)
    .chain(s.match_indices('\n').map(|(i, _)| i + 1))
    .collect::<Vec<_>>();
  line_offsets
    .iter()
    .enumerate()
    .map(|(i, offset)| {
      if i < line_offsets.len() - 1 {
        (*offset, line_offsets[i + 1] - offset - 1)
      } else {
        (*offset, s.len() - offset)
      }
    })
    .collect::<Vec<_>>()
}
