use std::collections::*;

use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;

pub fn compute_stable_retain<Prov: Provenance>(
  index: usize,
  to_add: &mut DynamicCollection<Prov>,
  stable_elem: &mut DynamicElement<Prov>,
  to_remove_to_add_indices: &mut HashSet<usize>,
  ctx: &Prov,
) -> bool {
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
    // If the two elements are equal, we will keep the element in stable if it is saturated
    if compute_saturation_and_update_tag(to_add_elem, stable_elem, ctx) {
      to_remove_to_add_indices.insert(index.clone());
      true
    } else {
      false
    }
  }
}

/// Compute whether the tag is saturated given a new and old version of the same element.
///
/// Note: Assume that the two elements are of the same tuple
pub fn compute_saturation_and_update_tag<Prov: Provenance>(
  to_add_elem: &mut DynamicElement<Prov>,
  stable_elem: &mut DynamicElement<Prov>,
  ctx: &Prov,
) -> bool {
  // If the two elements are equal, then we need to compute a new tag, while deciding where
  // to put the new element: stable or recent
  let new_tag = ctx.add(&stable_elem.tag, &to_add_elem.tag);
  let saturated = ctx.saturated(&stable_elem.tag, &new_tag);
  if saturated {
    // If we put the new element in stable, then we retain the stable element and update
    // the tag of that element. Additionally, we will remove the element in the `to_add`
    // collection.
    stable_elem.tag = new_tag;
    true
  } else {
    // If we put the new element in recent, then we will not retain the stable element,
    // while updating tag of the element in `to_add`
    to_add_elem.tag = new_tag;
    false
  }
}
