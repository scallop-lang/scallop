use std::cell::RefCell;
use std::rc::Rc;

use super::dataflow::*;
use super::*;
use crate::common::input_tag::*;
use crate::common::tuple::Tuple;
use crate::runtime::env::*;
use crate::runtime::monitor::*;
use crate::runtime::provenance::*;

#[derive(Clone)]
pub struct DynamicRelation<Prov: Provenance> {
  /// The facts derived
  pub stable: Rc<RefCell<Vec<DynamicCollection<Prov>>>>,

  /// The facts derived in the previous iteration
  pub recent: Rc<RefCell<DynamicCollection<Prov>>>,

  /// The batches of facts to be added in the next iteration
  to_add: Rc<RefCell<Vec<DynamicCollection<Prov>>>>,

  /// The waitlisted facts.
  /// The order of the facts corresponds to the order where the fact is derived.
  pub waitlist: Rc<RefCell<Vec<DynamicElement<Prov>>>>,
}

impl<Prov: Provenance> DynamicRelation<Prov> {
  pub fn new() -> Self {
    Self {
      stable: Rc::new(RefCell::new(Vec::new())),
      recent: Rc::new(RefCell::new(DynamicCollection::empty())),
      to_add: Rc::new(RefCell::new(Vec::new())),
      waitlist: Rc::new(RefCell::new(Vec::new())),
    }
  }

  pub fn insert_untagged<Tup>(&self, ctx: &Prov, data: Vec<Tup>)
  where
    Tup: Into<Tuple>,
  {
    let elements = data.into_iter().map(|tuple| (None, tuple)).collect::<Vec<_>>();
    self.insert_tagged(ctx, elements);
  }

  pub fn insert_untagged_with_monitor<Tup, M>(&self, ctx: &Prov, data: Vec<Tup>, m: &M)
  where
    Tup: Into<Tuple>,
    M: Monitor<Prov>,
  {
    let elements = data.into_iter().map(|tuple| (None, tuple)).collect::<Vec<_>>();
    self.insert_tagged_with_monitor(ctx, elements, m);
  }

  pub fn insert_one_tagged<Tup>(&self, ctx: &Prov, input_tag: Option<InputTagOf<Prov>>, tuple: Tup)
  where
    Tup: Into<Tuple>,
  {
    self.insert_tagged(ctx, vec![(input_tag, tuple)]);
  }

  pub fn insert_one_tagged_with_monitor<Tup, M>(
    &self,
    ctx: &mut Prov,
    input_tag: Option<InputTagOf<Prov>>,
    tuple: Tup,
    m: &M,
  ) where
    Tup: Into<Tuple>,
    M: Monitor<Prov>,
  {
    self.insert_tagged_with_monitor(ctx, vec![(input_tag, tuple)], m);
  }

  pub fn insert_dynamically_tagged<Tup>(&self, ctx: &Prov, data: Vec<(DynamicInputTag, Tup)>)
  where
    Tup: Into<Tuple>,
  {
    let elements = data
      .into_iter()
      .map(|(tag, tup)| {
        let input_tag = StaticInputTag::from_dynamic_input_tag(&tag);
        (input_tag, tup)
      })
      .collect();
    self.insert_tagged(ctx, elements);
  }

  pub fn insert_dynamically_tagged_with_monitor<Tup, M>(&self, ctx: &Prov, data: Vec<(DynamicInputTag, Tup)>, m: &M)
  where
    Tup: Into<Tuple>,
    M: Monitor<Prov>,
  {
    let elements = data
      .into_iter()
      .map(|(tag, tup)| {
        let input_tag = StaticInputTag::from_dynamic_input_tag(&tag);
        (input_tag, tup)
      })
      .collect();
    self.insert_tagged_with_monitor(ctx, elements, m);
  }

  pub fn insert_tagged<Tup>(&self, ctx: &Prov, data: Vec<(Option<InputTagOf<Prov>>, Tup)>)
  where
    Tup: Into<Tuple>,
  {
    let elements = data
      .into_iter()
      .map(|(info, tuple)| DynamicElement::new(tuple.into(), ctx.tagging_optional_fn(info)))
      .collect::<Vec<_>>();

    self
      .to_add
      .borrow_mut()
      .push(DynamicCollection::from_vec(elements, ctx));
  }

  pub fn insert_tagged_with_monitor<Tup, M>(&self, ctx: &Prov, data: Vec<(Option<InputTagOf<Prov>>, Tup)>, m: &M)
  where
    Tup: Into<Tuple>,
    M: Monitor<Prov>,
  {
    let elements = data
      .into_iter()
      .map(|(input_tag, tuple)| {
        let tag = ctx.tagging_optional_fn(input_tag.clone());
        let tuple: Tuple = tuple.into();
        m.observe_tagging(&tuple.clone(), &input_tag, &tag);
        DynamicElement::new(tuple, tag)
      })
      .collect::<Vec<_>>();

    self
      .to_add
      .borrow_mut()
      .push(DynamicCollection::from_vec(elements, ctx));
  }

  pub fn num_stable(&self) -> usize {
    self.stable.borrow().iter().fold(0, |a, rela| a + rela.len())
  }

  pub fn num_recent(&self) -> usize {
    self.recent.borrow().len()
  }

  pub fn num_to_add(&self) -> usize {
    self.to_add.borrow().iter().fold(0, |a, rela| a + rela.len())
  }

  pub fn clear_stable(&mut self) {
    self.stable.borrow_mut().clear()
  }

  pub fn changed(&mut self, ctx: &Prov, scheduler: &Scheduler) -> bool {
    // 1. Merge self.recent into self.stable.
    if !self.recent.borrow().is_empty() {
      let mut recent = ::std::mem::replace(&mut (*self.recent.borrow_mut()), DynamicCollection::empty());
      while self.stable.borrow().last().map(|x| x.len() <= 2 * recent.len()) == Some(true) {
        let last = self.stable.borrow_mut().pop().unwrap();
        recent = recent.merge(last, ctx);
      }
      self.stable.borrow_mut().push(recent);
    }

    // 2. Condense self.to_add
    let mut to_add_batches = self.to_add.borrow_mut();
    let mut to_add = if let Some(mut to_add) = to_add_batches.pop() {
      while let Some(to_add_more) = to_add_batches.pop() {
        to_add = to_add.merge(to_add_more, ctx);
      }
      to_add
    } else {
      DynamicCollection::empty()
    };

    // 3. Move all the waitlist into to-add for global
    // Note: This operation is very sub-optimal. We should find better algorithm
    scheduler.schedule(
      &mut to_add,
      &mut self.waitlist.borrow_mut(),
      &mut self.stable.borrow_mut(),
      ctx,
    );
    *self.recent.borrow_mut() = to_add;

    // 4. Finally, we decide whether there is a change by looking at whether recent is non-empty
    !self.recent.borrow().is_empty()
  }

  pub fn insert_dataflow_recent<'a>(
    &'a self,
    ctx: &'a Prov,
    d: &DynamicDataflow<'a, Prov>,
    runtime: &'a RuntimeEnvironment,
  ) {
    for batch in d.iter_recent() {
      let data = if runtime.early_discard {
        batch.filter(move |e| !ctx.discard(&e.tag)).collect::<Vec<_>>()
      } else {
        batch.collect::<Vec<_>>()
      };
      self.to_add.borrow_mut().push(DynamicCollection::from_vec(data, ctx));
    }
  }

  pub fn insert_dataflow_stable<'a>(
    &'a self,
    ctx: &'a Prov,
    d: &DynamicDataflow<'a, Prov>,
    runtime: &'a RuntimeEnvironment,
  ) {
    for batch in d.iter_stable() {
      let data = if runtime.early_discard {
        batch.filter(move |e| !ctx.discard(&e.tag)).collect::<Vec<_>>()
      } else {
        batch.collect::<Vec<_>>()
      };
      self.to_add.borrow_mut().push(DynamicCollection::from_vec(data, ctx));
    }
  }

  pub fn complete(&self, ctx: &Prov) -> DynamicCollection<Prov> {
    assert!(self.recent.borrow().is_empty());
    assert!(self.to_add.borrow().is_empty());
    let mut result = DynamicCollection::empty();
    while let Some(batch) = self.stable.borrow_mut().pop() {
      result = result.merge(batch, ctx);
    }
    result
  }
}
