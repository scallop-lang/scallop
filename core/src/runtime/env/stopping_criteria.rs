/// A list of stopping criteria for runtime execution
#[derive(Clone, Debug)]
pub struct StoppingCriteria {
  pub criteria: Vec<StoppingCriterion>,
}

impl Default for StoppingCriteria {
  fn default() -> Self {
    Self {
      criteria: vec![StoppingCriterion::default()],
    }
  }
}

impl StoppingCriteria {
  pub fn with_iter_limit(mut self, iter_limit: &Option<usize>) -> Self {
    // If we have iter limit, then add the criterion to the list
    if let Some(limit) = iter_limit {
      let criterion = StoppingCriterion::iter_limit(*limit);
      self.criteria.push(criterion);
    }

    // Return self
    self
  }

  pub fn set_iter_limit(&mut self, k: usize) {
    for criterion in &mut self.criteria {
      match criterion {
        StoppingCriterion::IterationLimit { limit } => {
          *limit = k;
          return;
        }
        _ => {}
      }
    }

    self.criteria.push(StoppingCriterion::iter_limit(k));
  }

  pub fn get_iter_limit(&self) -> Option<usize> {
    for criterion in &self.criteria {
      match criterion {
        StoppingCriterion::IterationLimit { limit } => return Some(*limit),
        _ => {}
      }
    }
    None
  }

  pub fn with_stop_when_goal_non_empty(mut self, stop: bool) -> Self {
    if stop {
      self.criteria.push(StoppingCriterion::NonEmptyGoalRelation)
    }
    self
  }

  pub fn stop_when_goal_relation_non_empty(&self) -> bool {
    for criterion in &self.criteria {
      match criterion {
        StoppingCriterion::NonEmptyGoalRelation => return true,
        _ => {}
      }
    }
    false
  }

  pub fn remove_iter_limit(&mut self) {
    self.criteria.retain(|c| !c.is_iter_limit())
  }
}

/// A pre-defined set of stopping criteria in Scallop
#[derive(Clone, Debug)]
pub enum StoppingCriterion {
  /// The execution will stop when the newly derived
  /// delta is empty with all the facts having saturated
  /// tag
  LeastFixPointSaturation,

  /// The execution will stop when the goal relation
  /// contains at least one fact.
  NonEmptyGoalRelation,

  /// Iteration limit
  IterationLimit {
    /// The limit
    limit: usize,
  },
}

impl Default for StoppingCriterion {
  fn default() -> Self {
    Self::LeastFixPointSaturation
  }
}

impl StoppingCriterion {
  pub fn iter_limit(k: usize) -> Self {
    Self::IterationLimit { limit: k }
  }

  pub fn is_iter_limit(&self) -> bool {
    match self {
      Self::IterationLimit { .. } => true,
      _ => false,
    }
  }

  pub fn is_non_empty_goal_relation(&self) -> bool {
    match self {
      Self::NonEmptyGoalRelation => true,
      _ => false,
    }
  }
}
