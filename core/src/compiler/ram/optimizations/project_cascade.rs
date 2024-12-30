use super::*;

pub fn project_cascade(ram: &mut Program) -> bool {
  let mut changed = false;

  // Iterate through all the updates
  for stratum in &mut ram.strata {
    for update in &mut stratum.updates {
      changed |= project_cascade_on_dataflow(&mut update.dataflow);
    }
  }

  // Return whether this optimization has changed anything
  changed
}

fn project_cascade_on_dataflow(d0: &mut Dataflow) -> bool {
  match d0 {
    Dataflow::Union(d1, d2) => {
      let r1 = project_cascade_on_dataflow(&mut **d1);
      let r2 = project_cascade_on_dataflow(&mut **d2);
      r1 || r2
    }
    Dataflow::Join(d1, d2) => {
      let r1 = project_cascade_on_dataflow(&mut **d1);
      let r2 = project_cascade_on_dataflow(&mut **d2);
      r1 || r2
    }
    Dataflow::Intersect(d1, d2) => {
      let r1 = project_cascade_on_dataflow(&mut **d1);
      let r2 = project_cascade_on_dataflow(&mut **d2);
      r1 || r2
    }
    Dataflow::Product(d1, d2) => {
      let r1 = project_cascade_on_dataflow(&mut **d1);
      let r2 = project_cascade_on_dataflow(&mut **d2);
      r1 || r2
    }
    Dataflow::Antijoin(d1, d2) => {
      let r1 = project_cascade_on_dataflow(&mut **d1);
      let r2 = project_cascade_on_dataflow(&mut **d2);
      r1 || r2
    }
    Dataflow::Difference(d1, d2) => {
      let r1 = project_cascade_on_dataflow(&mut **d1);
      let r2 = project_cascade_on_dataflow(&mut **d2);
      r1 || r2
    }
    Dataflow::Project(d1, p1) => match &**d1 {
      Dataflow::Project(d2, p2) => {
        *d0 = Dataflow::Project(d2.clone(), p1.compose(p2));
        project_cascade_on_dataflow(d0);
        true
      }
      _ => project_cascade_on_dataflow(d1),
    },
    Dataflow::JoinIndexedVec(d, _) => project_cascade_on_dataflow(&mut **d),
    Dataflow::Filter(d, _) => project_cascade_on_dataflow(&mut **d),
    Dataflow::Find(d, _) => project_cascade_on_dataflow(&mut **d),
    Dataflow::Sorted(d) => project_cascade_on_dataflow(&mut **d),
    Dataflow::OverwriteOne(d) => project_cascade_on_dataflow(&mut **d),
    Dataflow::Exclusion(d, _) => project_cascade_on_dataflow(&mut **d),
    Dataflow::ForeignPredicateConstraint(d, _, _) => project_cascade_on_dataflow(&mut **d),
    Dataflow::ForeignPredicateJoin(d, _, _) => project_cascade_on_dataflow(&mut **d),
    Dataflow::ForeignPredicateGround(_, _)
    | Dataflow::Unit(_)
    | Dataflow::Relation(_)
    | Dataflow::Reduce(_)
    | Dataflow::UntaggedVec(_) => false,
  }
}
