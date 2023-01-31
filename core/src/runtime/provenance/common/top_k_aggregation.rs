use std::collections::*;

pub fn aggregate_top_k_helper<F>(num_elements: usize, k: usize, weight_fn: F) -> Vec<usize>
where
  F: Fn(usize) -> f64,
{
  #[derive(Clone, Debug)]
  struct Element {
    id: usize,
    weight: f64,
  }

  impl std::cmp::PartialEq for Element {
    fn eq(&self, other: &Self) -> bool {
      self.id == other.id
    }
  }

  impl std::cmp::Eq for Element {}

  impl std::cmp::PartialOrd for Element {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
      other.weight.partial_cmp(&self.weight)
    }
  }

  impl std::cmp::Ord for Element {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
      if let Some(ord) = other.weight.partial_cmp(&self.weight) {
        ord
      } else {
        other.id.cmp(&self.id)
      }
    }
  }

  // Create a min-heap
  let mut heap = BinaryHeap::new();

  // First insert k elements into the heap
  let size = k.min(num_elements);
  for id in 0..size {
    let elem = Element {
      id,
      weight: weight_fn(id),
    };
    heap.push(elem);
  }

  // Then iterate through all other elements
  if heap.len() > 0 {
    for id in size..num_elements {
      let elem = Element {
        id,
        weight: weight_fn(id),
      };
      let min_elem_in_heap = heap.peek().unwrap();
      if &elem < min_elem_in_heap {
        heap.pop();
        heap.push(elem);
      }
    }
  }

  // Return the list of ids in the heap
  heap.into_iter().map(|elem| elem.id).collect()
}

pub fn unweighted_aggregate_top_k_helper<T>(elements: Vec<T>, k: usize) -> Vec<T> {
  if elements.len() <= k {
    elements
  } else {
    elements.into_iter().take(k).collect()
  }
}
