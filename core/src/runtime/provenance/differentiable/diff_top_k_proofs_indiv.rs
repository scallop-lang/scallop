use itertools::Itertools;

use super::*;
use crate::runtime::dynamic::*;
use crate::runtime::statics::*;
use crate::utils::PointerFamily;

#[derive(Clone)]
pub struct OutputIndivDiffProb<T: Clone> {
  pub k: usize,
  pub proofs: Vec<Vec<(f64, bool, T)>>,
}

impl<T: Clone> std::fmt::Debug for OutputIndivDiffProb<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("{")?;
    for (i, p) in self.proofs.iter().enumerate() {
      f.write_str("{")?;
      for (j, (prob, s, _)) in p.iter().enumerate() {
        if *s {
          f.write_fmt(format_args!("Pos({})", prob))?;
        } else {
          f.write_fmt(format_args!("Neg({})", prob))?;
        }
        if j + 1 < p.len() {
          f.write_str(", ")?;
        }
      }
      f.write_str("}")?;
      if i + 1 < self.proofs.len() {
        f.write_str(", ")?;
      }
    }
    f.write_str("}")
  }
}

impl<T: Clone> std::fmt::Display for OutputIndivDiffProb<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("{")?;
    for (i, p) in self.proofs.iter().enumerate() {
      f.write_str("{")?;
      for (j, (prob, s, _)) in p.iter().enumerate() {
        if *s {
          f.write_fmt(format_args!("Pos({})", prob))?;
        } else {
          f.write_fmt(format_args!("Neg({})", prob))?;
        }
        if j + 1 < p.len() {
          f.write_str(", ")?;
        }
      }
      f.write_str("}")?;
      if i + 1 < self.proofs.len() {
        f.write_str(", ")?;
      }
    }
    f.write_str("}")
  }
}

pub struct DiffTopKProofsIndivProvenance<T: Clone, P: PointerFamily> {
  pub k: usize,
  pub diff_probs: P::Pointer<Vec<(f64, T)>>,
  pub disjunctions: Disjunctions,
}

impl<T: Clone, P: PointerFamily> Clone for DiffTopKProofsIndivProvenance<T, P> {
  fn clone(&self) -> Self {
    Self {
      k: self.k,
      diff_probs: P::new((&*self.diff_probs).clone()),
      disjunctions: self.disjunctions.clone(),
    }
  }
}

impl<T: Clone, P: PointerFamily> DiffTopKProofsIndivProvenance<T, P> {
  pub fn new(k: usize) -> Self {
    Self {
      k,
      diff_probs: P::new(Vec::new()),
      disjunctions: Disjunctions::new(),
    }
  }

  pub fn input_tags(&self) -> Vec<T> {
    self.diff_probs.iter().map(|(_, t)| t.clone()).collect()
  }

  pub fn input_tag_of_fact_id(&self, i: usize) -> T {
    self.diff_probs[i].1.clone()
  }

  pub fn set_k(&mut self, k: usize) {
    self.k = k;
  }
}

impl<T: Clone, P: PointerFamily> DNFContextTrait for DiffTopKProofsIndivProvenance<T, P> {
  fn fact_probability(&self, id: &usize) -> f64 {
    self.diff_probs[*id].0
  }

  fn has_disjunction_conflict(&self, pos_facts: &std::collections::BTreeSet<usize>) -> bool {
    self.disjunctions.has_conflict(pos_facts)
  }
}

impl<T: Clone + 'static, P: PointerFamily> Provenance for DiffTopKProofsIndivProvenance<T, P> {
  type Tag = DNFFormula;

  type InputTag = InputExclusiveDiffProb<T>;

  type OutputTag = OutputIndivDiffProb<T>;

  fn name() -> &'static str {
    "diff-top-k-proofs-indiv"
  }

  fn tagging_fn(&mut self, input_tag: Self::InputTag) -> Self::Tag {
    let InputExclusiveDiffProb { prob, tag, exclusion } = input_tag;

    // First store the probability and generate the id
    let fact_id = self.diff_probs.len();
    P::get_mut(&mut self.diff_probs).push((prob, tag));

    // Store the mutual exclusivity
    if let Some(disjunction_id) = exclusion {
      self.disjunctions.add_disjunction(disjunction_id, fact_id);
    }

    // Finally return the formula
    DNFFormula::singleton(fact_id)
  }

  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    let k = self.k;
    let proofs = t
      .clauses
      .iter()
      .map(|clause| {
        clause
          .literals
          .iter()
          .map(|literal| {
            let fact_id = literal.fact_id();
            (
              self.fact_probability(&fact_id),
              literal.sign(),
              self.input_tag_of_fact_id(fact_id),
            )
          })
          .collect::<Vec<_>>()
      })
      .collect::<Vec<_>>();
    Self::OutputTag { k, proofs }
  }

  fn discard(&self, t: &Self::Tag) -> bool {
    t.is_empty()
  }

  fn zero(&self) -> Self::Tag {
    self.base_zero()
  }

  fn one(&self) -> Self::Tag {
    self.base_one()
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    self.top_k_add(t1, t2, self.k)
  }

  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool {
    t_old == t_new
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    self.top_k_mult(t1, t2, self.k)
  }

  fn negate(&self, t: &Self::Tag) -> Option<Self::Tag> {
    Some(self.top_k_negate(t, self.k))
  }

  fn weight(&self, t: &Self::Tag) -> f64 {
    let s = RealSemiring::new();
    let v = |i: &usize| self.diff_probs[i.clone()].0;
    t.wmc(&s, &v)
  }

  fn dynamic_count(&self, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    if batch.is_empty() {
      vec![DynamicElement::new(0usize, self.one())]
    } else {
      let mut elems = vec![];
      for chosen_set in (0..batch.len()).powerset() {
        let count = chosen_set.len();
        let tag = self.top_k_tag_of_chosen_set(batch.iter().map(|e| &e.tag), &chosen_set, self.k);
        elems.push(DynamicElement::new(count, tag));
      }
      elems
    }
  }

  fn dynamic_min(&self, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    let mut elems = vec![];
    for i in 0..batch.len() {
      let min_elem = batch[i].tuple.clone();
      let mut agg_tag = self.one();
      for j in 0..i {
        agg_tag = self.mult(&agg_tag, &self.negate(&batch[j].tag).unwrap());
      }
      agg_tag = self.mult(&agg_tag, &batch[i].tag);
      elems.push(DynamicElement::new(min_elem, agg_tag));
    }
    elems
  }

  fn dynamic_max(&self, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    let mut elems = vec![];
    for i in 0..batch.len() {
      let max_elem = batch[i].tuple.clone();
      let mut agg_tag = batch[i].tag.clone();
      for j in i + 1..batch.len() {
        agg_tag = self.mult(&agg_tag, &self.negate(&batch[j].tag).unwrap());
      }
      elems.push(DynamicElement::new(max_elem, agg_tag));
    }
    elems
  }

  fn dynamic_exists(&self, batch: DynamicElements<Self>) -> DynamicElements<Self> {
    let mut exists_tag = self.zero();
    let mut not_exists_tag = self.one();
    for elem in batch {
      exists_tag = self.add(&exists_tag, &elem.tag);
      not_exists_tag = self.mult(&not_exists_tag, &self.negate(&elem.tag).unwrap());
    }
    let t = DynamicElement::new(true, exists_tag);
    let f = DynamicElement::new(false, not_exists_tag);
    vec![t, f]
  }

  fn static_count<Tup: StaticTupleTrait>(&self, batch: StaticElements<Tup, Self>) -> StaticElements<usize, Self> {
    if batch.is_empty() {
      vec![StaticElement::new(0, self.one())]
    } else {
      let mut elems = vec![];
      for chosen_set in (0..batch.len()).powerset() {
        let count = chosen_set.len();
        let tag = self.top_k_tag_of_chosen_set(batch.iter().map(|e| &e.tag), &chosen_set, self.k);
        elems.push(StaticElement::new(count, tag));
      }
      elems
    }
  }

  fn static_min<Tup: StaticTupleTrait>(&self, batch: StaticElements<Tup, Self>) -> StaticElements<Tup, Self> {
    let mut elems = vec![];
    for i in 0..batch.len() {
      let min_elem = batch[i].tuple.get().clone();
      let mut agg_tag = self.one();
      for j in 0..i {
        agg_tag = self.mult(&agg_tag, &self.negate(&batch[j].tag).unwrap());
      }
      agg_tag = self.mult(&agg_tag, &batch[i].tag);
      elems.push(StaticElement::new(min_elem, agg_tag));
    }
    elems
  }

  fn static_max<Tup: StaticTupleTrait>(&self, batch: StaticElements<Tup, Self>) -> StaticElements<Tup, Self> {
    let mut elems = vec![];
    for i in 0..batch.len() {
      let max_elem = batch[i].tuple.get().clone();
      let mut agg_tag = batch[i].tag.clone();
      for j in i + 1..batch.len() {
        agg_tag = self.mult(&agg_tag, &self.negate(&batch[j].tag).unwrap());
      }
      elems.push(StaticElement::new(max_elem, agg_tag));
    }
    elems
  }

  fn static_exists<Tup: StaticTupleTrait>(&self, batch: StaticElements<Tup, Self>) -> StaticElements<bool, Self> {
    let mut exists_tag = self.zero();
    let mut not_exists_tag = self.one();
    for elem in batch {
      exists_tag = self.add(&exists_tag, &elem.tag);
      not_exists_tag = self.mult(&not_exists_tag, &self.negate(&elem.tag).unwrap());
    }
    let t = StaticElement::new(true, exists_tag);
    let f = StaticElement::new(false, not_exists_tag);
    vec![t, f]
  }
}
