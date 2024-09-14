use std::collections::*;

use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;

use super::schedulers;

#[derive(Clone, Debug)]
pub struct SchedulerManager {
  pub base_scheduler: Scheduler,
  pub relation_scheduler: HashMap<String, Scheduler>,
}

impl Default for SchedulerManager {
  fn default() -> Self {
    SchedulerManager {
      base_scheduler: Scheduler::default(),
      relation_scheduler: HashMap::new(),
    }
  }
}

impl SchedulerManager {
  pub fn new_with_default_scheduler(scheduler: Scheduler) -> Self {
    SchedulerManager {
      base_scheduler: scheduler,
      relation_scheduler: HashMap::new(),
    }
  }

  pub fn get_scheduler(&self, relation: &String) -> &Scheduler {
    if let Some(sche) = self.relation_scheduler.get(relation) {
      sche
    } else {
      &self.base_scheduler
    }
  }

  pub fn get_default_scheduler(&self) -> &Scheduler {
    &self.base_scheduler
  }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Scheduler {
  /// Least Fixed-Point (Breadth-first Search Algorithm)
  LFP,

  /// Depth-first Search Algorithm
  DFS,

  /// A-Star Search Algorithm
  AStar,

  /// Beam Search Algorithm with beam size
  Beam {
    beam_size: usize
  },
}

impl Scheduler {
  pub fn from_args(scheduler: Option<String>, beam_size: Option<usize>) -> Result<Option<Self>, String> {
    if let Some(scheduler) = scheduler {
      match scheduler.as_str() {
        "lfp" | "bfs" => {
          Ok(Some(Self::LFP))
        },
        "dfs" => {
          Ok(Some(Self::DFS))
        },
        "a-star" | "astar" => {
          Ok(Some(Self::AStar))
        },
        "beam" => {
          Ok(Some(Self::Beam { beam_size: beam_size.unwrap_or(3) }))
        },
        s => {
          Err(format!("Unknown scheduler `{s}`"))
        }
      }
    } else {
      Ok(None)
    }
  }

  pub fn schedule<'a, Prov: Provenance>(
    &self,
    delta: &'a mut DynamicCollection<Prov>,
    waitlist: &'a mut Vec<DynamicElement<Prov>>,
    stable: &'a mut Vec<DynamicCollection<Prov>>,
    ctx: &'a Prov,
  ) {
    match self {
      Self::LFP => schedulers::lfp::schedule_lfp(delta, waitlist, stable, ctx),
      Self::DFS => schedulers::dfs::schedule_dfs(delta, waitlist, stable, ctx),
      Self::AStar => schedulers::astar::schedule_a_star(delta, waitlist, stable, ctx),
      Self::Beam { beam_size } => schedulers::beam::schedule_beam(delta, waitlist, stable, ctx, *beam_size),
    }
  }
}

impl Default for Scheduler {
  fn default() -> Self {
    Self::LFP
  }
}
