use std::collections::*;

use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;
use crate::runtime::utils::*;

use super::utils::{compute_saturation_and_update_tag, compute_stable_retain};

pub fn schedule_dfs<'a, Prov: Provenance>(
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

  // If there exists at least one element left in the to_add set
  if to_add.len() > 0 {
    // Then we leave only one element in to_add and leave everything else to waitlist
    let num_elems = to_add.len();
    let mut drainer = to_add.drain();
    let first = drainer.next().unwrap(); // Unwrap will go through since delta is not empty

    // Insert everything into waitlist
    waitlist.reserve(num_elems - 1); // num_elems - 1 is okay since there is at least one elem
    waitlist.extend(drainer);

    // For the first element, we add that back to delta
    to_add.insert_unchecked(first);
  } else if waitlist.len() > 0 {
    // Otherwise, we will need to find at least one element in waitlist that is non-saturated
    // For any element in the waitlist that is saturated (that we compute along the way), we
    // update the stable batch with the new index already
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
}
