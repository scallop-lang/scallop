pub fn cartesian(sizes: Vec<usize>) -> Vec<Vec<usize>> {
  struct CartesianIterator {
    total: Vec<usize>,
    indices: Vec<usize>,
    has_curr: bool,
  }

  impl CartesianIterator {
    pub fn new(total: Vec<usize>) -> Self {
      let indices = std::iter::repeat(0).take(total.len()).collect();
      let has_curr = total.iter().all(|s| *s > 0);
      Self {
        total,
        indices,
        has_curr,
      }
    }
  }

  impl Iterator for CartesianIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
      if self.has_curr {
        let result = self.indices.clone();
        let mut maybe_to_add_index = Some(0);
        while let Some(to_add_index) = maybe_to_add_index {
          if to_add_index >= self.indices.len() {
            self.has_curr = false;
            maybe_to_add_index = None;
          } else {
            let curr_index = self.indices[to_add_index];
            if curr_index + 1 < self.total[to_add_index] {
              self.indices[to_add_index] += 1;
              maybe_to_add_index = None;
            } else {
              self.indices[to_add_index] = 0;
              maybe_to_add_index = Some(to_add_index + 1)
            }
          }
        }
        Some(result)
      } else {
        None
      }
    }
  }

  CartesianIterator::new(sizes).collect()
}
