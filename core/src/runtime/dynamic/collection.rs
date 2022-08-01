use std::cmp::Ordering;

use super::*;
use crate::common::tuple::Tuple;
use crate::runtime::monitor::Monitor;
use crate::runtime::provenance::*;

#[derive(Clone)]
pub struct DynamicCollection<T: Tag> {
  pub elements: Vec<DynamicElement<T>>,
}

impl<T: Tag + std::fmt::Debug> std::fmt::Debug for DynamicCollection<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_set().entries(&self.elements).finish()
  }
}

impl<T: Tag> std::fmt::Display for DynamicCollection<T>
where
  DynamicElement<T>: std::fmt::Display,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("{")?;
    for (i, elem) in self.elements.iter().enumerate() {
      if i > 0 {
        f.write_str(", ")?;
      }
      std::fmt::Display::fmt(elem, f)?;
    }
    f.write_str("}")
  }
}

impl<T: Tag> DynamicCollection<T> {
  pub fn empty() -> Self {
    Self { elements: vec![] }
  }

  pub fn from_vec_unchecked(elements: Vec<DynamicElement<T>>) -> Self {
    Self { elements }
  }

  pub fn from_vec(mut elements: Vec<DynamicElement<T>>, ctx: &T::Context) -> Self {
    elements.sort();

    let mut index = 0;
    let mut to_keep = 0;
    if elements.len() > 1 {
      let mut last_index = index;
      to_keep += 1;
      index += 1;
      while index < elements.len() {
        if &elements[index].tuple == &elements[last_index].tuple {
          let new_tag = ctx.add(&elements[last_index].tag, &elements[index].tag);
          elements[last_index].tag = new_tag;
        } else {
          if to_keep < index {
            elements.swap(to_keep, index);
          }
          last_index = to_keep;
          to_keep += 1;
        }
        index += 1;
      }
      elements.truncate(to_keep);
    }

    Self { elements }
  }

  pub fn len(&self) -> usize {
    self.elements.len()
  }

  pub fn is_empty(&self) -> bool {
    self.elements.is_empty()
  }

  pub fn ith(&self, i: usize) -> Option<&DynamicElement<T>> {
    self.elements.get(i)
  }

  pub fn iter(&self) -> impl Iterator<Item = &DynamicElement<T>> {
    self.elements.iter()
  }

  pub fn into_iter(self) -> impl IntoIterator<Item = DynamicElement<T>> {
    self.elements.into_iter()
  }

  pub fn apply_recover_fn<F, S>(self, mut f: F) -> impl Iterator<Item = (S, Tuple)>
  where
    F: FnMut(T) -> S,
  {
    self.elements.into_iter().map(move |elem| (f(elem.tag), elem.tuple))
  }

  pub fn recover(self, ctx: &T::Context) -> DynamicOutputCollection<T> {
    DynamicOutputCollection::from(
      self
        .elements
        .into_iter()
        .map(move |elem| (ctx.recover_fn(&elem.tag), elem.tuple)),
    )
  }

  pub fn recover_with_monitor<M>(self, ctx: &T::Context, m: &M) -> DynamicOutputCollection<T>
  where
    M: Monitor<T::Context>,
  {
    DynamicOutputCollection::from(self.elements.into_iter().map(move |elem| {
      let output_tag = ctx.recover_fn(&elem.tag);
      m.observe_recover(&elem.tuple, &elem.tag, &output_tag);
      (output_tag, elem.tuple)
    }))
  }

  pub fn merge(self, other: Self, ctx: &T::Context) -> Self {
    let Self {
      elements: mut elements1,
    } = self;
    let Self {
      elements: mut elements2,
    } = other;

    // If one of the element lists is zero-length, we don't need to do any work
    if elements1.is_empty() {
      return Self { elements: elements2 };
    }

    if elements2.is_empty() {
      return Self { elements: elements1 };
    }

    // Make sure that elements1 starts with the lower element
    // Will not panic since both collections must have at least 1 element at this point
    if elements1[0] > elements2[0] {
      std::mem::swap(&mut elements1, &mut elements2);
    }

    // Fast path for when all the new elements are after the exiting ones
    if elements1[elements1.len() - 1] < elements2[0] {
      elements1.extend(elements2.into_iter());
      return Self { elements: elements1 };
    }

    let mut elements = Vec::with_capacity(elements1.len() + elements2.len());
    let mut elements1 = elements1.drain(..);
    let mut elements2 = elements2.drain(..).peekable();

    elements.push(elements1.next().unwrap());
    if elements.first() == elements2.peek() {
      let e2 = elements2.next().unwrap();
      elements[0].tag = ctx.add(&elements[0].tag, &e2.tag);
    }

    for mut elem in elements1 {
      while elements2.peek().map(|x| x.cmp(&elem)) == Some(Ordering::Less) {
        elements.push(elements2.next().unwrap());
      }
      if elements2.peek().map(|x| x.cmp(&elem)) == Some(Ordering::Equal) {
        // Merge the tags
        let e2 = elements2.peek().unwrap();
        elem.tag = ctx.add(&elem.tag, &e2.tag);

        elements2.next();
      }
      elements.push(elem);
    }

    // Finish draining second list
    elements.extend(elements2);

    Self { elements }
  }
}

impl<T: Tag> std::ops::Deref for DynamicCollection<T> {
  type Target = [DynamicElement<T>];

  fn deref(&self) -> &Self::Target {
    &self.elements[..]
  }
}

impl<T: Tag> std::ops::DerefMut for DynamicCollection<T> {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.elements[..]
  }
}

impl<T: Tag> std::ops::Index<usize> for DynamicCollection<T> {
  type Output = DynamicElement<T>;

  fn index(&self, index: usize) -> &Self::Output {
    &self.elements[index]
  }
}

impl<T: Tag> std::ops::IndexMut<usize> for DynamicCollection<T> {
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    &mut self.elements[index]
  }
}
