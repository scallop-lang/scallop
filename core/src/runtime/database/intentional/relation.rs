use crate::runtime::dynamic::{DynamicCollection, DynamicOutputCollection};
use crate::runtime::monitor::Monitor;
use crate::runtime::provenance::*;
use crate::utils::PointerFamily;

pub struct IntentionalRelation<Prov: Provenance, Ptr: PointerFamily> {
  /// Recovered
  pub recovered: bool,

  /// Internal facts
  pub internal_facts: DynamicCollection<Prov>,

  /// Recovered facts
  pub recovered_facts: Ptr::Rc<DynamicOutputCollection<Prov>>,
}

impl<Prov: Provenance, Ptr: PointerFamily> Default for IntentionalRelation<Prov, Ptr> {
  fn default() -> Self {
    Self::new()
  }
}

impl<Prov: Provenance, Ptr: PointerFamily> Clone for IntentionalRelation<Prov, Ptr> {
  fn clone(&self) -> Self {
    Self {
      recovered: self.recovered,
      internal_facts: self.internal_facts.clone(),
      recovered_facts: Ptr::clone_rc(&self.recovered_facts),
    }
  }
}

impl<Prov: Provenance, Ptr: PointerFamily> std::fmt::Debug for IntentionalRelation<Prov, Ptr> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("IDBRelation")
      .field("internal", &self.internal_facts)
      .field("recovered", &Ptr::get_rc(&self.recovered_facts))
      .finish()
  }
}

impl<Prov: Provenance, Ptr: PointerFamily> std::fmt::Display for IntentionalRelation<Prov, Ptr> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    Ptr::get_rc(&self.recovered_facts).fmt(f)
  }
}

impl<Prov: Provenance, Ptr: PointerFamily> IntentionalRelation<Prov, Ptr> {
  pub fn new() -> Self {
    Self {
      recovered: false,
      internal_facts: DynamicCollection::empty(),
      recovered_facts: Ptr::new_rc(DynamicOutputCollection::empty()),
    }
  }

  pub fn from_dynamic_collection(collection: DynamicCollection<Prov>) -> Self {
    Self {
      recovered: false,
      internal_facts: collection,
      recovered_facts: Ptr::new_rc(DynamicOutputCollection::empty()),
    }
  }

  pub fn from_dynamic_output_collection(collection: DynamicOutputCollection<Prov>) -> Self {
    Self {
      recovered: true,
      internal_facts: DynamicCollection::empty(),
      recovered_facts: Ptr::new_rc(collection),
    }
  }

  pub fn recover_with_monitor<M: Monitor<Prov>>(&mut self, ctx: &Prov, m: &M, drain: bool) {
    // Only recover if it is not recovered
    if !self.recovered && !self.internal_facts.is_empty() {
      if drain {
        // Add internal facts to recovered facts, and remove the internal facts
        Ptr::get_rc_mut(&mut self.recovered_facts).extend(self.internal_facts.drain().map(|elem| {
          let output_tag = ctx.recover_fn(&elem.tag);
          m.observe_recover(&elem.tuple, &elem.tag, &output_tag);
          (output_tag, elem.tuple)
        }));
      } else {
        // Add internal facts to recover facts, do not remove the internal facts
        Ptr::get_rc_mut(&mut self.recovered_facts).extend(self.internal_facts.iter().map(|elem| {
          let output_tag = ctx.recover_fn(&elem.tag);
          m.observe_recover(&elem.tuple, &elem.tag, &output_tag);
          (output_tag, elem.tuple.clone())
        }));
      }

      // Set recovered to be true
      self.recovered = true;
    }
  }

  pub fn recover(&mut self, ctx: &Prov, drain: bool) {
    // Only recover if it is not recovered
    if !self.recovered {
      // Shortcut: if there is no internal facts, then there is nothing to recover
      if self.internal_facts.is_empty() {
        self.recovered = true;
        return;
      }

      // Check if we need to drain the internal facts
      if drain {
        // Add internal facts to recovered facts, and remove the internal facts
        Ptr::get_rc_mut(&mut self.recovered_facts).extend(self.internal_facts.drain().map(|elem| {
          let output_tag = ctx.recover_fn(&elem.tag);
          (output_tag, elem.tuple)
        }));
      } else {
        // Add internal facts to recover facts, do not remove the internal facts
        Ptr::get_rc_mut(&mut self.recovered_facts).extend(self.internal_facts.iter().map(|elem| {
          let output_tag = ctx.recover_fn(&elem.tag);
          (output_tag, elem.tuple.clone())
        }));
      }

      // Set recovered to be true
      self.recovered = true;
    }
  }
}
