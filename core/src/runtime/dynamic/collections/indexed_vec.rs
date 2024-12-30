use super::*;
use crate::common::tuple::Tuple;
use crate::common::value_type::ValueType;
use crate::runtime::provenance::*;

#[derive(Clone)]
pub struct DynamicIndexedVecCollection<Prov: Provenance> {
  index_type: ValueType,
  vec: Vec<Option<(Tuple, Prov::Tag)>>,
  length: usize,
}

impl<Prov: Provenance> Into<DynamicCollection<Prov>> for DynamicIndexedVecCollection<Prov> {
  fn into(self) -> DynamicCollection<Prov> {
    DynamicCollection::IndexedVec(self)
  }
}

impl<Prov: Provenance> DynamicIndexedVecCollection<Prov> {
  pub fn empty() -> Self {
    Self {
      index_type: ValueType::USize,
      vec: vec![],
      length: 0,
    }
  }

  pub fn from_vec(elements: Vec<DynamicElement<Prov>>, ctx: &Prov) -> Self {
    if elements.is_empty() {
      Self {
        index_type: ValueType::USize,
        vec: vec![],
        length: 0,
      }
    } else {
      let first_tuple = &elements[0].tuple;
      if let Some(index_type) = first_tuple[0].value_type() {
        // Initialize a storage and reserve a capacity
        let mut vec = Vec::new();
        vec.reserve(elements.len());

        // Initialize a length
        let mut length = 0;

        // Insert elements into the storage
        for elem in elements {
          let DynamicElement { tuple, tag } = elem;
          let mut values = tuple.as_values();

          // Obtain the head and tail
          let head = values.remove(0);
          let tail = if values.len() == 1 {
            Tuple::Value(values[0].clone())
          } else {
            Tuple::from_values(values.into_iter())
          };
          let index = head.cast_to_usize();

          // Allocate content correspondingly
          if index + 1 > vec.len() {
            vec.resize(index + 1, None);
          }

          // Put the element into
          if let Some((existing_tup, old_tag)) = &mut vec[index] {
            if existing_tup == &tail {
              // Merge the tags
              *old_tag = ctx.add(old_tag, &tag);
            } else {
              // Use later element to overwrite previous elements
              *existing_tup = tail;
              *old_tag = tag;
            }
          } else {
            vec[index] = Some((tail, tag));
            length += 1;
          }
        }

        // Generate the collection
        Self {
          index_type,
          vec,
          length,
        }
      } else {
        panic!("Cannot obtain the index type from a tuple {}", first_tuple);
      }
    }
  }

  pub fn len(&self) -> usize {
    self.length
  }

  pub fn is_empty(&self) -> bool {
    self.length == 0
  }

  /// Get an element from the indexed vector; this operation is $O(1)$.
  pub fn get(&self, index: usize) -> Option<(Tuple, Prov::Tag)> {
    self.vec.get(index).and_then(|e| e.clone())
  }

  pub fn iter<'a>(&'a self) -> DynamicIndexedVecCollectionIter<'a, Prov> {
    DynamicIndexedVecCollectionIter {
      index_type: self.index_type.clone(),
      vec_iter: self.vec.iter().enumerate(),
    }
  }

  pub fn into_iter(self) -> DynamicIndexedVecCollectionIntoIter<Prov> {
    DynamicIndexedVecCollectionIntoIter {
      index_type: self.index_type.clone(),
      vec_iter: self.vec.into_iter().enumerate(),
    }
  }

  pub fn drain<'a>(&'a mut self) -> DynamicIndexedVecCollectionDrainer<'a, Prov> {
    DynamicIndexedVecCollectionDrainer {
      index_type: self.index_type.clone(),
      vec_drainer: self.vec.drain(..).enumerate()
    }
  }
}

impl<Prov: Provenance> std::fmt::Debug for DynamicIndexedVecCollection<Prov>
where
  Prov::Tag: std::fmt::Debug,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_set().entries(self.iter()).finish()
  }
}

impl<Prov: Provenance> std::fmt::Display for DynamicIndexedVecCollection<Prov>
where
  DynamicElement<Prov>: std::fmt::Display,
{
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str("{")?;
    for (index, elem) in self.iter().enumerate() {
      if index > 0 {
        f.write_str(", ")?;
      }
      std::fmt::Display::fmt(&elem, f)?;
    }
    f.write_str("}")
  }
}

#[derive(Clone)]
pub struct DynamicIndexedVecCollectionIter<'a, Prov: Provenance> {
  index_type: ValueType,
  vec_iter: std::iter::Enumerate<core::slice::Iter<'a, Option<(Tuple, Prov::Tag)>>>,
}

impl<'a, Prov: Provenance> Iterator for DynamicIndexedVecCollectionIter<'a, Prov> {
  type Item = DynamicElement<Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    while let Some((index, curr_elem)) = self.vec_iter.next() {
      if let Some((tail_tup, tag)) = curr_elem {
        return Some(construct_dyn_elem(&self.index_type, index, tail_tup.clone(), tag.clone()))
      }
    }
    None
  }
}

pub struct DynamicIndexedVecCollectionIntoIter<Prov: Provenance> {
  index_type: ValueType,
  vec_iter: std::iter::Enumerate<std::vec::IntoIter<Option<(Tuple, Prov::Tag)>>>,
}

impl<Prov: Provenance> Iterator for DynamicIndexedVecCollectionIntoIter<Prov> {
  type Item = DynamicElement<Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    while let Some((index, curr_elem)) = self.vec_iter.next() {
      if let Some((tail_tup, tag)) = curr_elem {
        return Some(construct_dyn_elem(&self.index_type, index, tail_tup, tag))
      }
    }
    None
  }
}

pub struct DynamicIndexedVecCollectionDrainer<'a, Prov: Provenance> {
  index_type: ValueType,
  vec_drainer: std::iter::Enumerate<std::vec::Drain<'a, Option<(Tuple, Prov::Tag)>>>,
}

impl<'a, Prov: Provenance> Iterator for DynamicIndexedVecCollectionDrainer<'a, Prov> {
  type Item = DynamicElement<Prov>;

  fn next(&mut self) -> Option<Self::Item> {
    while let Some((index, curr_elem)) = self.vec_drainer.next() {
      if let Some((tail_tup, tag)) = curr_elem {
        return Some(construct_dyn_elem(&self.index_type, index, tail_tup, tag))
      }
    }
    None
  }
}

fn construct_dyn_elem<Prov: Provenance>(
  index_type: &ValueType,
  index: usize,
  tail: Tuple,
  tag: Prov::Tag,
) -> DynamicElement<Prov> {
  let index_value = index_type.value_from_usize(index).expect("Expect a value type that can be casted from usize");
  let tup = Tuple::from_values(std::iter::once(index_value).chain(tail.in_order_values()));
  let complete_elem = DynamicElement::<Prov>::new(tup, tag.clone());
  return complete_elem
}
