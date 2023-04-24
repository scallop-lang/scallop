use std::collections::*;

use crate::runtime::database::extensional::ExtensionalRelation;
use crate::runtime::dynamic::*;
use crate::runtime::monitor::*;
use crate::runtime::provenance::*;
use crate::utils::*;

use super::*;

#[derive(Debug, Clone)]
pub struct IntentionalDatabase<Prov: Provenance, Ptr: PointerFamily = RcFamily> {
  /// Intentional relations
  pub intentional_relations: BTreeMap<String, IntentionalRelation<Prov, Ptr>>,
}

impl<Prov: Provenance, Ptr: PointerFamily> Default for IntentionalDatabase<Prov, Ptr> {
  fn default() -> Self {
    Self::new()
  }
}

impl<Prov: Provenance, Ptr: PointerFamily> IntentionalDatabase<Prov, Ptr> {
  pub fn new() -> Self {
    Self {
      intentional_relations: BTreeMap::new(),
    }
  }

  /// Clone the intentional database into a new one with a different provenance.
  /// The new database will be empty.
  pub fn clone_with_new_provenance<Prov2: Provenance>(&self) -> IntentionalDatabase<Prov2, Ptr> {
    IntentionalDatabase::new()
  }

  /// Create an intentional database from dynamic collections
  pub fn from_dynamic_collections<I>(iter: I) -> Self
  where
    I: Iterator<Item = (String, DynamicCollection<Prov>)>,
  {
    let intentional_relations = iter
      .map(|(relation_name, collection)| {
        let relation = IntentionalRelation::from_dynamic_collection(collection);
        (relation_name, relation)
      })
      .collect();
    Self { intentional_relations }
  }

  /// Extend intentional relation
  pub fn extend<I>(&mut self, iter: I)
  where
    I: Iterator<Item = (String, IntentionalRelation<Prov, Ptr>)>,
  {
    self.intentional_relations.extend(iter)
  }

  pub fn insert_dynamic_collection(&mut self, relation: String, collection: DynamicCollection<Prov>) {
    self
      .intentional_relations
      .insert(relation, IntentionalRelation::from_dynamic_collection(collection));
  }

  /// Insert dynamic output collection
  pub fn insert_dynamic_output_collection(&mut self, relation: String, collection: DynamicOutputCollection<Prov>) {
    self.intentional_relations.insert(
      relation,
      IntentionalRelation::from_dynamic_output_collection(collection),
    );
  }

  /// Check if the database contains this relation (regardless of whether recovered or not)
  pub fn has_relation(&self, relation: &str) -> bool {
    self.intentional_relations.contains_key(relation)
  }

  pub fn recover_from_edb(&mut self, relation: &str, ctx: &Prov, edb_relation: &ExtensionalRelation<Prov>) {
    self.intentional_relations.insert(
      relation.to_string(),
      IntentionalRelation {
        recovered: true,
        internal_facts: DynamicCollection::empty(),
        recovered_facts: Ptr::new_rc(DynamicOutputCollection::from(
          edb_relation
            .internal
            .iter()
            .map(|elem| (ctx.recover_fn(&elem.tag), elem.tuple.clone())),
        )),
      },
    );
  }

  /// Recover the output collection for a relation
  pub fn recover(&mut self, relation: &str, ctx: &Prov, drain: bool) {
    if let Some(r) = self.intentional_relations.get_mut(relation) {
      r.recover(ctx, drain);
    }
  }

  /// Recover the output collection for a relation, with a monitor
  pub fn recover_with_monitor<M: Monitor<Prov>>(&mut self, relation: &str, ctx: &Prov, m: &M, drain: bool) {
    if let Some(r) = self.intentional_relations.get_mut(relation) {
      // !SPECIAL MONITORING!
      m.observe_recovering_relation(relation);
      r.recover_with_monitor(ctx, m, drain);
    }
  }

  /// Get internal collection
  pub fn get_internal_collection(&self, relation: &str) -> Option<&DynamicCollection<Prov>> {
    self.intentional_relations.get(relation).map(|r| &r.internal_facts)
  }

  /// Get recovered collection
  pub fn get_output_collection_ref(&self, relation: &str) -> Option<&DynamicOutputCollection<Prov>> {
    self.intentional_relations.get(relation).and_then(|r| {
      if r.recovered {
        Some(Ptr::get_rc(&r.recovered_facts))
      } else {
        None
      }
    })
  }

  pub fn get_output_collection(&self, relation: &str) -> Option<Ptr::Rc<DynamicOutputCollection<Prov>>> {
    self.intentional_relations.get(relation).and_then(|r| {
      if r.recovered {
        Some(Ptr::clone_rc(&r.recovered_facts))
      } else {
        None
      }
    })
  }

  /// Retain the results under the given relations and remove everything else
  pub fn retain_relations(&mut self, relations: &HashSet<String>) {
    self.intentional_relations.retain(|k, _| relations.contains(k))
  }

  pub fn remove_relation(&mut self, relation: &str) {
    self.intentional_relations.remove(relation);
  }

  pub fn remove_relations(&mut self, relations: &HashSet<String>) {
    self.intentional_relations.retain(|k, _| !relations.contains(k));
  }
}

impl<Prov: Provenance, Ptr: PointerFamily> IntoIterator for IntentionalDatabase<Prov, Ptr> {
  type Item = (String, IntentionalRelation<Prov, Ptr>);

  type IntoIter = std::collections::btree_map::IntoIter<String, IntentionalRelation<Prov, Ptr>>;

  fn into_iter(self) -> Self::IntoIter {
    self.intentional_relations.into_iter()
  }
}

impl<'a, Prov: Provenance, Ptr: PointerFamily> IntoIterator for &'a IntentionalDatabase<Prov, Ptr> {
  type Item = (&'a String, &'a IntentionalRelation<Prov, Ptr>);

  type IntoIter = std::collections::btree_map::Iter<'a, String, IntentionalRelation<Prov, Ptr>>;

  fn into_iter(self) -> Self::IntoIter {
    self.intentional_relations.iter()
  }
}

impl<'a, Prov: Provenance, Ptr: PointerFamily> IntoIterator for &'a mut IntentionalDatabase<Prov, Ptr> {
  type Item = (&'a String, &'a mut IntentionalRelation<Prov, Ptr>);

  type IntoIter = std::collections::btree_map::IterMut<'a, String, IntentionalRelation<Prov, Ptr>>;

  fn into_iter(self) -> Self::IntoIter {
    self.intentional_relations.iter_mut()
  }
}
