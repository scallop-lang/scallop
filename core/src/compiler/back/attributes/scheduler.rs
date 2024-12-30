use crate::runtime::env::Scheduler;

use super::*;

#[derive(Debug, Clone)]
pub struct SchedulerAttribute {
  pub scheduler: Scheduler,
}

impl SchedulerAttribute {
  pub fn new(scheduler: Scheduler) -> Self {
    Self { scheduler }
  }
}

impl AttributeTrait for SchedulerAttribute {
  fn name(&self) -> String {
    "scheduler".to_string()
  }

  fn args(&self) -> Vec<String> {
    match &self.scheduler {
      Scheduler::LFP => {
        vec!["\"lfp\"".to_string()]
      },
      Scheduler::AStar => {
        vec!["\"a-star\"".to_string()]
      },
      Scheduler::DFS => {
        vec!["\"dfs\"".to_string()]
      },
      Scheduler::Beam { beam_size } => {
        vec![
          "\"beam\"".to_string(),
          format!("beam_size={beam_size}"),
        ]
      },
    }
  }
}
