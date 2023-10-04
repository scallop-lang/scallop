use std::collections::*;
use std::env;
use std::fs;

use crate::common::foreign_tensor::*;
use crate::common::tuple::*;
use crate::runtime::provenance::*;
use crate::utils::*;

use super::*;

#[derive(Clone)]
pub struct DumpProofsInternal {
  current_tagging_relation: Option<String>,
  tagged_tuples: HashMap<String, Vec<(usize, Option<f64>, String)>>,
  current_recovering_relation: Option<String>,
  recovered_tuples: HashMap<String, Vec<(String, f64, Vec<Vec<(bool, usize)>>)>>,
}

pub struct DumpProofsMonitor {
  internal: <ArcFamily as PointerFamily>::RcCell<DumpProofsInternal>,
}

impl DumpProofsMonitor {
  pub fn new() -> Self {
    Self {
      internal: <ArcFamily as PointerFamily>::new_rc_cell(DumpProofsInternal {
        current_tagging_relation: None,
        tagged_tuples: HashMap::new(),
        current_recovering_relation: None,
        recovered_tuples: HashMap::new(),
      }),
    }
  }

  pub fn set_current_tagging_relation(&self, relation: &str) {
    ArcFamily::get_rc_cell_mut(&self.internal, |c| {
      c.current_tagging_relation = Some(relation.to_string());
    })
  }

  pub fn record_tag(&self, tup: &Tuple, prob: Option<f64>, tag: &DNFFormula) {
    ArcFamily::get_rc_cell_mut(&self.internal, |c| {
      if let Some(r) = &c.current_tagging_relation {
        if let Some(id) = tag.get_singleton_id() {
          c.tagged_tuples
            .entry(r.to_string())
            .or_default()
            .push((id, prob, tup.to_string()));
        }
      }
    })
  }

  pub fn set_current_recovering_relation(&self, relation: &str) {
    ArcFamily::get_rc_cell_mut(&self.internal, |c| {
      c.current_recovering_relation = Some(relation.to_string());
    })
  }

  pub fn record_recover(&self, tup: &Tuple, prob: f64, tag: &DNFFormula) {
    ArcFamily::get_rc_cell_mut(&self.internal, |c| {
      if let Some(r) = &c.current_recovering_relation {
        let proofs = tag
          .clauses
          .iter()
          .map(|clause| {
            clause
              .literals
              .iter()
              .map(|literal| match literal {
                Literal::Pos(id) => (true, *id),
                Literal::Neg(id) => (false, *id),
              })
              .collect::<Vec<_>>()
          })
          .collect::<Vec<_>>();
        c.recovered_tuples
          .entry(r.to_string())
          .or_default()
          .push((tup.to_string(), prob, proofs));
      }
    })
  }

  pub fn dump_relation(&self, relation: &str) {
    ArcFamily::get_rc_cell_mut(&self.internal, |c| {
      let dir = env::var("SCALLOP_DUMP_PROOFS_DIR").unwrap_or(".tmp/dumped-tuples".to_string());
      for (relation, tuples) in &c.tagged_tuples {
        let js = serde_json::to_string(&tuples).expect("Cannot serialize tuples");
        fs::write(&format!("{dir}/{relation}.json"), js).expect("Unable to write file");
      }
      if let Some(tuples) = c.recovered_tuples.get(relation) {
        let js = serde_json::to_string(&tuples).expect("Cannot serialize tuples");
        fs::write(&format!("{dir}/{relation}.json"), js).expect("Unable to write file");
      }
    })
  }
}

impl Clone for DumpProofsMonitor {
  fn clone(&self) -> Self {
    Self {
      internal: ArcFamily::get_rc_cell(&self.internal, |x| ArcFamily::new_rc_cell(x.clone())),
    }
  }
}

impl<Prov: Provenance> Monitor<Prov> for DumpProofsMonitor {
  default fn name(&self) -> &'static str {
    "dump-proofs"
  }
  default fn observe_finish_execution(&self) {}
  default fn observe_loading_relation(&self, _: &str) {}
  default fn observe_tagging(&self, _: &Tuple, _: &Option<Prov::InputTag>, _: &Prov::Tag) {}
  default fn observe_recovering_relation(&self, _: &str) {}
  default fn observe_recover(&self, _: &Tuple, _: &Prov::Tag, _: &Prov::OutputTag) {}
  default fn observe_finish_recovering_relation(&self, _: &str) {}
}

impl<P: PointerFamily> Monitor<top_k_proofs::TopKProofsProvenance<P>> for DumpProofsMonitor {
  fn name(&self) -> &'static str {
    "dump-proofs"
  }

  fn observe_loading_relation(&self, relation: &str) {
    self.set_current_tagging_relation(relation)
  }

  fn observe_tagging(&self, tup: &Tuple, input_tag: &Option<InputExclusiveProb>, tag: &DNFFormula) {
    self.record_tag(tup, input_tag.as_ref().map(|v| v.prob), tag)
  }

  fn observe_recovering_relation(&self, relation: &str) {
    self.set_current_recovering_relation(relation)
  }

  fn observe_recover(&self, tup: &Tuple, tag: &DNFFormula, output_tag: &f64) {
    self.record_recover(tup, output_tag.clone(), tag)
  }

  fn observe_finish_recovering_relation(&self, relation: &str) {
    self.dump_relation(relation)
  }
}

impl<T: FromTensor, P: PointerFamily> Monitor<diff_top_k_proofs::DiffTopKProofsProvenance<T, P>> for DumpProofsMonitor {
  fn name(&self) -> &'static str {
    "dump-proofs"
  }

  fn observe_loading_relation(&self, relation: &str) {
    self.set_current_tagging_relation(relation)
  }

  fn observe_tagging(&self, tup: &Tuple, input_tag: &Option<InputExclusiveDiffProb<T>>, tag: &DNFFormula) {
    self.record_tag(tup, input_tag.as_ref().map(|v| v.prob), tag)
  }

  fn observe_recovering_relation(&self, relation: &str) {
    self.set_current_recovering_relation(relation)
  }

  fn observe_recover(&self, tup: &Tuple, tag: &DNFFormula, output_tag: &OutputDiffProb) {
    self.record_recover(tup, output_tag.0.clone(), tag)
  }

  fn observe_finish_recovering_relation(&self, relation: &str) {
    self.dump_relation(relation)
  }
}
