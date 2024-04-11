use crate::utils::*;

/// The differentiable probability storage that offers interior mutability
pub struct DiffProbStorage<T: Clone, P: PointerFamily> {
  pub storage: P::RcCell<Vec<(f64, Option<T>)>>,
  pub num_requires_grad: P::Cell<usize>,
}

impl<T: Clone, P: PointerFamily> DiffProbStorage<T, P> {
  pub fn new() -> Self {
    Self {
      storage: P::new_rc_cell(Vec::new()),
      num_requires_grad: P::new_cell(0),
    }
  }

  pub fn new_with_placeholder() -> Self {
    Self {
      storage: P::new_rc_cell(Vec::new()),
      num_requires_grad: P::new_cell(1),
    }
  }

  /// Clone the internal storage
  pub fn clone_internal(&self) -> Self {
    Self {
      storage: P::new_rc_cell(P::get_rc_cell(&self.storage, |s| s.clone())),
      num_requires_grad: P::clone_cell(&self.num_requires_grad),
    }
  }

  /// Clone the reference counter
  pub fn clone_rc(&self) -> Self {
    Self {
      storage: P::clone_rc_cell(&self.storage),
      num_requires_grad: P::clone_cell(&self.num_requires_grad),
    }
  }

  pub fn add_prob(&self, prob: f64, external_tag: Option<T>) -> usize {
    // Store the fact id
    let fact_id = P::get_rc_cell(&self.storage, |s| s.len());

    // Increment the `num_requires_grad` if the external tag is provided
    if external_tag.is_some() {
      P::get_cell_mut(&self.num_requires_grad, |n| *n += 1);
    }

    // Push this element into the storage
    P::get_rc_cell_mut(&self.storage, |s| s.push((prob, external_tag)));

    // Return the id
    fact_id
  }

  pub fn add_prob_with_id(&self, prob: f64, external_tag: Option<T>, id: usize) -> usize {
    // Increment the `num_requires_grad` if the external tag is provided
    if external_tag.is_some() {
      P::get_cell_mut(&self.num_requires_grad, |n| *n += 1);
    }

    // Add
    P::get_rc_cell_mut(&self.storage, |s| {
      if id >= s.len() {
        s.extend(std::iter::repeat_n((0.0, None), id - s.len() + 1));
      }

      s[id] = (prob.clone(), external_tag.clone())
    });

    // Return the id
    id
  }

  pub fn get_diff_prob(&self, id: &usize) -> (f64, Option<T>) {
    P::get_rc_cell(&self.storage, |d| d[id.clone()].clone())
  }

  pub fn get_prob(&self, id: &usize) -> f64 {
    P::get_rc_cell(&self.storage, |d| d[id.clone()].0)
  }

  pub fn input_tags(&self) -> Vec<T> {
    P::get_rc_cell(&self.storage, |s| s.iter().filter_map(|(_, t)| t.clone()).collect())
  }

  pub fn num_input_tags(&self) -> usize {
    P::get_cell(&self.num_requires_grad, |i| *i)
  }

  pub fn fact_probability(&self, id: &usize) -> f64 {
    P::get_rc_cell(&self.storage, |d| d[*id].0)
  }
}

impl<T: Clone, P: PointerFamily> Clone for DiffProbStorage<T, P> {
  /// Clone the reference counter of this storage (shallow copy)
  fn clone(&self) -> Self {
    self.clone_rc()
  }
}
