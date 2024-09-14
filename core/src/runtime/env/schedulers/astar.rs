///! A-Star Search Scheduling
use std::collections::*;

use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;
use crate::runtime::utils::*;

use super::utils::{compute_saturation_and_update_tag, compute_stable_retain};

/// Assumption: waitlist is sorted by weight from small to large
pub fn schedule_a_star<'a, Prov: Provenance>(
  to_add: &'a mut DynamicCollection<Prov>,
  waitlist: &'a mut Vec<DynamicElement<Prov>>,
  stable: &'a mut Vec<DynamicCollection<Prov>>,
  ctx: &'a Prov,
) {
  // First go over the to_add set
  if to_add.len() > 0 {
    let mut to_remove_to_add_indices = HashSet::new();
    for stable_batch in stable.iter_mut() {
      let mut index = 0;

      // Go over the stable batch and retain the related elements
      if to_add.len() > 4 * stable_batch.len() {
        stable_batch.elements.retain_mut(|stable_elem| {
          index = gallop_index(to_add, index, |y| y < stable_elem);
          compute_stable_retain(index, to_add, stable_elem, &mut to_remove_to_add_indices, ctx)
        });
      } else {
        stable_batch.elements.retain_mut(|stable_elem| {
          while index < to_add.len() && &to_add[index] < stable_elem {
            index += 1;
          }
          compute_stable_retain(index, to_add, stable_elem, &mut to_remove_to_add_indices, ctx)
        });
      }
    }

    // Remove the elements in `to_add` that we deem not needed
    to_add.retain(|i, _| !to_remove_to_add_indices.contains(&i));
  }

  // Drain to_add and merge it into waitlist. This preserves the order of elements in the waitlist
  for to_add_elem in to_add.drain() {
    // Store the weight
    let to_add_elem_weight = ctx.weight(&to_add_elem.tag);

    // Check if we can insert to_add_element in the middle of waitlist
    let to_insert_index = gallop_index(&waitlist, 0, |elem| ctx.weight(&elem.tag) < to_add_elem_weight);
    if let Some(existing_elem) = waitlist.get_mut(to_insert_index) {
      if &existing_elem.tuple == &to_add_elem.tuple {
        existing_elem.tag = ctx.add(&existing_elem.tag, &to_add_elem.tag);
      } else {
        waitlist.insert(to_insert_index, to_add_elem);
      }
    } else {
      waitlist.push(to_add_elem);
    }
  }

  // First go over the to_add set
  while let Some(mut waitlist_elem) = waitlist.pop() {
    // Check if the element exists in any of the stable batch
    let mut not_found = true;
    for stable_batch in stable.iter_mut() {
      let stable_index = gallop_index(&stable_batch.elements, 0, |y| y < &waitlist_elem);
      if stable_index >= stable_batch.len() || &stable_batch[stable_index].tuple != &waitlist_elem.tuple {
        // This element is not found in the current stable batch
      } else {
        let stable_elem = &mut stable_batch[stable_index];
        if compute_saturation_and_update_tag(&mut waitlist_elem, stable_elem, ctx) {
          // If saturated, then the waitlisted item does not count. We want to remove this element from waitlist
          not_found = false;
          break;
        } else {
          // If not saturated, we need to remove the element from the stable
          stable_batch.retain(|id, _| id != stable_index);

          // We add the element to to_add
          to_add.insert_unchecked(waitlist_elem);

          // We found an element. Return
          return;
        }
      }
    }

    // Check not found
    if not_found {
      to_add.insert_unchecked(waitlist_elem);
      return;
    }
  }
}
