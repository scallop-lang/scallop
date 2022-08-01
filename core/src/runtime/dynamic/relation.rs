use std::cell::RefCell;
use std::collections::*;
use std::rc::Rc;

use super::dataflow::*;
use super::*;
use crate::common::input_tag::FromInputTag;
use crate::common::input_tag::InputTag;
use crate::common::tuple::Tuple;
use crate::runtime::monitor::*;
use crate::runtime::provenance::*;
use crate::runtime::utils::*;

#[derive(Clone)]
pub struct DynamicRelation<T: Tag> {
  pub stable: Rc<RefCell<Vec<DynamicCollection<T>>>>,
  pub recent: Rc<RefCell<DynamicCollection<T>>>,
  to_add: Rc<RefCell<Vec<DynamicCollection<T>>>>,
}

impl<T: Tag> DynamicRelation<T> {
  pub fn new() -> Self {
    Self {
      stable: Rc::new(RefCell::new(Vec::new())),
      recent: Rc::new(RefCell::new(DynamicCollection::empty())),
      to_add: Rc::new(RefCell::new(Vec::new())),
    }
  }

  pub fn insert_untagged<Tup>(&self, ctx: &mut T::Context, data: Vec<Tup>)
  where
    Tup: Into<Tuple>,
  {
    let elements = data.into_iter().map(|tuple| (None, tuple)).collect::<Vec<_>>();
    self.insert_tagged(ctx, elements);
  }

  pub fn insert_untagged_with_monitor<Tup, M>(&self, ctx: &mut T::Context, data: Vec<Tup>, m: &M)
  where
    Tup: Into<Tuple>,
    M: Monitor<T::Context>,
  {
    let elements = data.into_iter().map(|tuple| (None, tuple)).collect::<Vec<_>>();
    self.insert_tagged_with_monitor(ctx, elements, m);
  }

  pub fn insert_one_tagged<Tup>(&self, ctx: &mut T::Context, input_tag: Option<InputTagOf<T::Context>>, tuple: Tup)
  where
    Tup: Into<Tuple>,
  {
    self.insert_tagged(ctx, vec![(input_tag, tuple)]);
  }

  pub fn insert_one_tagged_with_monitor<Tup, M>(
    &self,
    ctx: &mut T::Context,
    input_tag: Option<InputTagOf<T::Context>>,
    tuple: Tup,
    m: &M,
  ) where
    Tup: Into<Tuple>,
    M: Monitor<T::Context>,
  {
    self.insert_tagged_with_monitor(ctx, vec![(input_tag, tuple)], m);
  }

  pub fn insert_dynamically_tagged<Tup>(&self, ctx: &mut T::Context, data: Vec<(InputTag, Tup)>)
  where
    Tup: Into<Tuple>,
  {
    let elements = data
      .into_iter()
      .map(|(tag, tup)| {
        let input_tag = FromInputTag::from_input_tag(&tag);
        (input_tag, tup)
      })
      .collect();
    self.insert_tagged(ctx, elements);
  }

  pub fn insert_dynamically_tagged_with_monitor<Tup, M>(&self, ctx: &mut T::Context, data: Vec<(InputTag, Tup)>, m: &M)
  where
    Tup: Into<Tuple>,
    M: Monitor<T::Context>,
  {
    let elements = data
      .into_iter()
      .map(|(tag, tup)| {
        let input_tag = FromInputTag::from_input_tag(&tag);
        (input_tag, tup)
      })
      .collect();
    self.insert_tagged_with_monitor(ctx, elements, m);
  }

  pub fn insert_tagged<Tup>(&self, ctx: &mut T::Context, data: Vec<(Option<InputTagOf<T::Context>>, Tup)>)
  where
    Tup: Into<Tuple>,
  {
    let elements = data
      .into_iter()
      .map(|(info, tuple)| DynamicElement::new(tuple.into(), ctx.tagging_optional_fn(info)))
      .collect::<Vec<_>>();
    let dataflow = DynamicDataflow::Vec(&elements);
    self.insert_dataflow_recent(ctx, &dataflow);
  }

  pub fn insert_tagged_with_monitor<Tup, M>(
    &self,
    ctx: &mut T::Context,
    data: Vec<(Option<InputTagOf<T::Context>>, Tup)>,
    m: &M,
  ) where
    Tup: Into<Tuple>,
    M: Monitor<T::Context>,
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
    let dataflow = DynamicDataflow::Vec(&elements);
    self.insert_dataflow_recent(ctx, &dataflow);
  }

  pub fn insert_annotated_disjunction<Tup>(
    &self,
    ctx: &mut T::Context,
    data: Vec<(Option<InputTagOf<T::Context>>, Tup)>,
  ) where
    Tup: Into<Tuple>,
  {
    let (base_infos, tuples): (Vec<_>, Vec<Tup>) = data.into_iter().unzip();
    let tags = ctx.tagging_disjunction_optional_fn(base_infos);
    let new_data = tags
      .into_iter()
      .zip(tuples.into_iter().map(|t| t.into()))
      .map(|(tag, tup)| DynamicElement::new(tup, tag))
      .collect::<Vec<_>>();
    let dataflow = DynamicDataflow::Vec(&new_data);
    self.insert_dataflow_recent(ctx, &dataflow);
  }

  pub fn insert_annotated_disjunction_with_monitor<Tup, M>(
    &self,
    ctx: &mut T::Context,
    data: Vec<(Option<InputTagOf<T::Context>>, Tup)>,
    m: &M,
  ) where
    Tup: Into<Tuple>,
    M: Monitor<T::Context>,
  {
    let (input_tags, tuples): (Vec<_>, Vec<Tup>) = data.into_iter().unzip();
    let tags = ctx.tagging_disjunction_optional_fn(input_tags.clone());
    let elems = tuples
      .into_iter()
      .map(|t| t.into())
      .zip(input_tags.into_iter())
      .zip(tags.into_iter())
      .map(|((tup, input_tag), tag)| {
        m.observe_tagging(&tup, &input_tag, &tag);
        DynamicElement::new(tup, tag)
      })
      .collect::<Vec<_>>();
    let dataflow = DynamicDataflow::Vec(&elems);
    self.insert_dataflow_recent(ctx, &dataflow);
  }

  pub fn num_stable(&self) -> usize {
    self.stable.borrow().iter().fold(0, |a, rela| a + rela.len())
  }

  pub fn num_recent(&self) -> usize {
    self.recent.borrow().len()
  }

  pub fn changed(&mut self, ctx: &T::Context) -> bool {
    // 1. Merge self.recent into self.stable.
    if !self.recent.borrow().is_empty() {
      let mut recent = ::std::mem::replace(&mut (*self.recent.borrow_mut()), DynamicCollection::empty());
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
           to_add: &mut DynamicCollection<T>,
           stable_elem: &mut DynamicElement<T>,
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

  pub fn insert_dataflow_recent<'a>(&self, ctx: &T::Context, d: &DynamicDataflow<'a, T>) {
    for batch in d.iter_recent() {
      let data = batch.filter(|e| !ctx.discard(&e.tag)).collect::<Vec<_>>();
      self.to_add.borrow_mut().push(DynamicCollection::from_vec(data, ctx));
    }
  }

  pub fn insert_dataflow_stable<'a>(&self, ctx: &T::Context, d: &DynamicDataflow<'a, T>) {
    for batch in d.iter_stable() {
      let data = batch.filter(|e| ctx.discard(&e.tag)).collect::<Vec<_>>();
      self.to_add.borrow_mut().push(DynamicCollection::from_vec(data, ctx));
    }
  }

  pub fn complete(&self, ctx: &T::Context) -> DynamicCollection<T> {
    assert!(self.recent.borrow().is_empty());
    assert!(self.to_add.borrow().is_empty());
    let mut result = DynamicCollection::empty();
    while let Some(batch) = self.stable.borrow_mut().pop() {
      result = result.merge(batch, ctx);
    }
    result
  }
}
