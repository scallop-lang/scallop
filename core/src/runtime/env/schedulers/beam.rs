///! A-Star Search Scheduling
use std::collections::*;

use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;
use crate::runtime::utils::*;

use super::utils::compute_stable_retain;

/// Assumption: waitlist is sorted by weight from small to large
pub fn schedule_beam<'a, Prov: Provenance>(
  to_add: &'a mut DynamicCollection<Prov>,
  waitlist: &'a mut Vec<DynamicElement<Prov>>,
  stable: &'a mut Vec<DynamicCollection<Prov>>,
  ctx: &'a Prov,
  beam_size: usize,
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

  // Drain to_add and sort them
  let mut sorted_delta = Vec::<DynamicElement<Prov>>::new();
  for to_add_elem in to_add.drain() {
    // Store the weight
    let to_add_elem_weight = ctx.weight(&to_add_elem.tag);

    // Check if we can insert to_add_element in the middle of waitlist
    let to_insert_index = gallop_index(&sorted_delta, 0, |elem| ctx.weight(&elem.tag) < to_add_elem_weight);
    sorted_delta.insert(to_insert_index, to_add_elem);
  }

  // If there exists at least beam_size element in to_add
  if sorted_delta.len() >= beam_size {
    // Add the last elements in the delta to to_add
    for _ in 0..beam_size {
      to_add.insert_unchecked(sorted_delta.pop().unwrap());
    }

    // The rest goes into waitlist; notice that this is in chronological + weight-increasing
    // manner
    waitlist.extend(sorted_delta.into_iter());
  } else {
    // We first insert everything from the delta to the to_add
    for elem in sorted_delta {
      to_add.insert_unchecked(elem);
    }

    // If there is not enough elements in delta, we will need to pull from
    // the waitlist. We need to pop somethings from the waitlist. We can just pop the last ones
    while to_add.len() < beam_size && waitlist.len() > 0 {
      to_add.insert(waitlist.pop().unwrap(), ctx);
    }
  }
}
