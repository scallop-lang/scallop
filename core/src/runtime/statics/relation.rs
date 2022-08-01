use std::cell::RefCell;
use std::collections::*;
use std::rc::Rc;

use super::dataflow::Dataflow;
use super::*;
use crate::common::tuple::{AsTuple, Tuple};
use crate::runtime::edb::*;
use crate::runtime::monitor::Monitor;
use crate::runtime::provenance::*;
use crate::runtime::utils::gallop_index;

#[derive(Clone)]
pub struct StaticRelation<Tup: StaticTupleTrait, T: Tag> {
  pub stable: Rc<RefCell<Vec<StaticCollection<Tup, T>>>>,
  pub recent: Rc<RefCell<StaticCollection<Tup, T>>>,
  to_add: Rc<RefCell<Vec<StaticCollection<Tup, T>>>>,
}

impl<Tup: StaticTupleTrait, T: Tag> StaticRelation<Tup, T> {
  pub fn new() -> Self {
    Self {
      stable: Rc::new(RefCell::new(Vec::new())),
      recent: Rc::new(RefCell::new(StaticCollection::empty())),
      to_add: Rc::new(RefCell::new(Vec::new())),
    }
  }

  pub fn insert_untagged(&self, ctx: &mut T::Context, v: Vec<Tup>) {
    let data = v.into_iter().map(|tuple| (None, tuple)).collect::<Vec<_>>();
    self.insert_tagged(ctx, data)
  }

  pub fn insert_untagged_with_monitor<M>(&self, ctx: &mut T::Context, v: Vec<Tup>, m: &M)
  where
    Tuple: From<Tup>,
    M: Monitor<T::Context>,
    InputTagOf<T::Context>: std::fmt::Debug,
  {
    let data = v.into_iter().map(|tuple| (None, tuple)).collect::<Vec<_>>();
    self.insert_tagged_with_monitor(ctx, data, m)
  }

  pub fn insert_one_tagged(&self, ctx: &mut T::Context, info: Option<InputTagOf<T::Context>>, tuple: Tup) {
    self.insert_tagged(ctx, vec![(info, tuple)]);
  }

  pub fn insert_one_with_input_tag_and_monitor<M>(
    &self,
    ctx: &mut T::Context,
    info: Option<InputTagOf<T::Context>>,
    tuple: Tup,
    m: &M,
  ) where
    Tuple: From<Tup>,
    M: Monitor<T::Context>,
    InputTagOf<T::Context>: std::fmt::Debug,
  {
    self.insert_tagged_with_monitor(ctx, vec![(info, tuple)], m);
  }

  pub fn insert_from_edb(&self, ctx: &mut T::Context, relation: EDBRelation<T::Context>)
  where
    Tuple: AsTuple<Tup>,
  {
    let EDBRelation { facts, disjunctions } = relation;

    // Collect disjunctions
    let mut disj_i = HashSet::new();
    disjunctions.into_iter().for_each(|disj| {
      let disj_facts = disj
        .into_iter()
        .map(|i| {
          disj_i.insert(i);
          let f = facts[i].clone();
          (f.tag, <Tuple as AsTuple<Tup>>::as_tuple(&f.tuple))
        })
        .collect::<Vec<_>>();
      self.insert_annotated_disjunction(ctx, disj_facts);
    });

    // Collect non-disjuntion facts
    let non_disj_facts = facts
      .into_iter()
      .enumerate()
      .filter(|(i, _)| !disj_i.contains(i))
      .map(|(_, f)| (f.tag, <Tuple as AsTuple<Tup>>::as_tuple(&f.tuple)))
      .collect::<Vec<_>>();
    self.insert_tagged(ctx, non_disj_facts);
  }

  pub fn insert_from_edb_with_monitor<M>(&self, ctx: &mut T::Context, relation: EDBRelation<T::Context>, m: &M)
  where
    Tuple: AsTuple<Tup> + From<Tup>,
    M: Monitor<T::Context>,
    InputTagOf<T::Context>: std::fmt::Debug,
  {
    let EDBRelation { facts, disjunctions } = relation;

    // Collect disjunctions
    let mut disj_i = HashSet::new();
    disjunctions.into_iter().for_each(|disj| {
      let disj_facts = disj
        .into_iter()
        .map(|i| {
          disj_i.insert(i);
          let f = facts[i].clone();
          (f.tag, <Tuple as AsTuple<Tup>>::as_tuple(&f.tuple))
        })
        .collect::<Vec<_>>();
      self.insert_annotated_disjunction_with_monitor(ctx, disj_facts, m);
    });

    // Collect non-disjuntion facts
    let non_disj_facts = facts
      .into_iter()
      .enumerate()
      .filter(|(i, _)| !disj_i.contains(i))
      .map(|(_, f)| (f.tag, <Tuple as AsTuple<Tup>>::as_tuple(&f.tuple)))
      .collect::<Vec<_>>();
    self.insert_tagged_with_monitor(ctx, non_disj_facts, m);
  }

  pub fn insert_tagged(&self, ctx: &mut T::Context, data: Vec<(Option<InputTagOf<T::Context>>, Tup)>) {
    let elements = data
      .into_iter()
      .map(|(input_tag, tuple)| StaticElement::new(tuple, ctx.tagging_optional_fn(input_tag)))
      .collect::<Vec<_>>();
    self.insert_dataflow_recent(ctx, elements);
  }

  pub fn insert_tagged_with_monitor<M>(
    &self,
    ctx: &mut T::Context,
    data: Vec<(Option<InputTagOf<T::Context>>, Tup)>,
    m: &M,
  ) where
    Tuple: From<Tup>,
    M: Monitor<T::Context>,
    InputTagOf<T::Context>: std::fmt::Debug,
  {
    let elements = data
      .into_iter()
      .map(|(input_tag, tuple)| {
        let tag = ctx.tagging_optional_fn(input_tag.clone());
        m.observe_tagging(&tuple.clone().into(), &input_tag, &tag);
        StaticElement::new(tuple, tag)
      })
      .collect::<Vec<_>>();
    self.insert_dataflow_recent(ctx, elements);
  }

  pub fn insert_annotated_disjunction(&self, ctx: &mut T::Context, facts: Vec<(Option<InputTagOf<T::Context>>, Tup)>) {
    let (tags, tuples) = facts.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
    let internal_tags = ctx.tagging_disjunction_optional_fn(tags);
    let elems = tuples
      .into_iter()
      .zip(internal_tags.into_iter())
      .map(|(tup, tag)| StaticElement::new(tup, tag))
      .collect::<Vec<_>>();
    self.insert_dataflow_recent(ctx, elems);
  }

  pub fn insert_annotated_disjunction_with_monitor<M>(
    &self,
    ctx: &mut T::Context,
    facts: Vec<(Option<InputTagOf<T::Context>>, Tup)>,
    m: &M,
  ) where
    Tuple: From<Tup>,
    M: Monitor<T::Context>,
    InputTagOf<T::Context>: std::fmt::Debug,
  {
    let (input_tags, tuples) = facts.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
    let tags = ctx.tagging_disjunction_optional_fn(input_tags.clone());
    let elems = tuples
      .into_iter()
      .zip(input_tags.into_iter())
      .zip(tags.into_iter())
      .map(|((tup, input_tag), tag)| {
        m.observe_tagging(&tup.clone().into(), &input_tag, &tag);
        StaticElement::new(tup, tag)
      })
      .collect::<Vec<_>>();
    self.insert_dataflow_recent(ctx, elems);
  }

  // pub fn insert_dynamic_with_input_tag(
  //   &self,
  //   ctx: &mut T::Context,
  //   data: Vec<(Option<InputTagOf<T::Context>>, Tuple)>,
  // ) where
  //   Tuple: AsTuple<Tup>,
  // {
  //   let data = data
  //     .into_iter()
  //     .map(|(tag, tup)| StaticElement::new(<Tuple as AsTuple<Tup>>::as_tuple(&tup), ctx.tagging_optional_fn(tag)))
  //     .collect::<Vec<_>>();
  //   self.insert_dataflow_recent(ctx, data)
  // }

  pub fn insert_dynamic_elements(&self, ctx: &mut T::Context, data: Vec<crate::runtime::dynamic::DynamicElement<T>>)
  where
    Tuple: AsTuple<Tup>,
  {
    let data = data
      .into_iter()
      .map(|e| StaticElement::new(<Tuple as AsTuple<Tup>>::as_tuple(&e.tuple), e.tag.clone()))
      .collect::<Vec<_>>();
    self.insert_dataflow_recent(ctx, data)
  }

  pub fn num_stable(&self) -> usize {
    self.stable.borrow().iter().map(|rela| rela.len()).sum()
  }

  pub fn num_recent(&self) -> usize {
    self.recent.borrow().len()
  }

  pub fn insert_dataflow_recent<D>(&self, ctx: &T::Context, d: D)
  where
    D: Dataflow<Tup, T>,
  {
    for batch in d.iter_recent() {
      let data = batch.filter(|e| !ctx.discard(&e.tag)).collect::<Vec<_>>();
      self.to_add.borrow_mut().push(StaticCollection::from_vec(data, ctx));
    }
  }

  pub fn insert_dataflow_stable<D>(&self, ctx: &T::Context, d: D)
  where
    D: Dataflow<Tup, T>,
  {
    for batch in d.iter_stable() {
      let data = batch.filter(|e| !ctx.discard(&e.tag)).collect::<Vec<_>>();
      self.to_add.borrow_mut().push(StaticCollection::from_vec(data, ctx));
    }
  }

  pub fn complete(&self, ctx: &T::Context) -> StaticCollection<Tup, T> {
    assert!(self.recent.borrow().is_empty());
    assert!(self.to_add.borrow().is_empty());
    let mut result = StaticCollection::empty();
    while let Some(batch) = self.stable.borrow_mut().pop() {
      result = result.merge(batch, ctx);
    }
    result
  }
}

pub trait StaticRelationTrait<T: Tag> {
  fn changed(&mut self, semiring_ctx: &T::Context) -> bool;
}

impl<Tup, T> StaticRelationTrait<T> for StaticRelation<Tup, T>
where
  Tup: StaticTupleTrait,
  T: Tag,
{
  fn changed(&mut self, ctx: &T::Context) -> bool {
    // 1. Merge self.recent into self.stable.
    if !self.recent.borrow().is_empty() {
      let mut recent = ::std::mem::replace(&mut (*self.recent.borrow_mut()), StaticCollection::empty());
      while self.stable.borrow().last().map(|x| x.len() <= 2 * recent.len()) == Some(true) {
        let last = self.stable.borrow_mut().pop().unwrap();
        recent = recent.merge(last, ctx);
      }
      self.stable.borrow_mut().push(recent);
    }

    // 2. Move self.to_add into self.recent.
    let to_add = self.to_add.borrow_mut().pop();
    if let Some(mut to_add) = to_add {
      let mut to_remove_to_add_indices = HashSet::new();
      while let Some(to_add_more) = self.to_add.borrow_mut().pop() {
        to_add = to_add.merge(to_add_more, ctx);
      }

      // Make sure that there is no duplicates; if there is, merge the tag back to the stable
      for stable_batch in self.stable.borrow_mut().iter_mut() {
        let mut index = 0;

        // Helper function to compute whether to retain a given stable element
        let compute_stable_retain =
          |index: usize,
           to_add: &mut StaticCollection<Tup, T>,
           stable_elem: &mut StaticElement<Tup, T>,
           to_remove_to_add_indices: &mut HashSet<usize>| {
            // If going over to_add, then the stable element does not exist in to_add. Therefore we retain the stable element
            if index >= to_add.len() {
              return true;
            }

            // Otherwise, we can safely access the index in to_add collection
            let to_add_elem = &mut to_add[index];
            if &to_add_elem.tuple != &stable_elem.tuple {
              // If the two elements are not equal, we retain the element in stable batch
              true
            } else {
              // If the two elements are equal, then we need to compute a new tag, while deciding where
              // to put the new element: stable or recent
              let (new_tag, proceeding) = ctx.add_with_proceeding(&stable_elem.tag, &to_add_elem.tag);
              match proceeding {
                Proceeding::Recent => {
                  // If we put the new element in recent, then we will not retain the stable element,
                  // while updating tag of the element in `to_add`
                  to_add_elem.tag = new_tag;
                  false
                }
                Proceeding::Stable => {
                  // If we put the new element in stable, then we retain the stable element and update
                  // the tag of that element. Additionally, we will remove the element in the `to_add`
                  // collection.
                  stable_elem.tag = new_tag;
                  to_remove_to_add_indices.insert(index.clone());
                  true
                }
              }
            }
          };

        // Go over the stable batch and retain the related elements
        if to_add.len() > 4 * stable_batch.len() {
          stable_batch.elements.retain_mut(|x| {
            index = gallop_index(&to_add, index, |y| y < x);
            compute_stable_retain(index, &mut to_add, x, &mut to_remove_to_add_indices)
          });
        } else {
          stable_batch.elements.retain_mut(|x| {
            while index < to_add.len() && &to_add[index] < x {
              index += 1;
            }
            compute_stable_retain(index, &mut to_add, x, &mut to_remove_to_add_indices)
          });
        }
      }

      // Remove the elements in `to_add` that we deem not needed
      to_add.elements = to_add
        .elements
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !to_remove_to_add_indices.contains(i))
        .map(|(_, e)| e)
        .collect();
      *self.recent.borrow_mut() = to_add;
    }

    !self.recent.borrow().is_empty()
  }
}
