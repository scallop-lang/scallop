use std::collections::*;

use crate::runtime::dynamic::*;
use crate::runtime::provenance::*;
use crate::runtime::utils::*;

use super::utils::compute_stable_retain;

pub fn schedule_lfp<'a, Prov: Provenance>(
  to_add: &'a mut DynamicCollection<Prov>,
  #[allow(unused)] waitlist: &'a mut Vec<DynamicElement<Prov>>,
  stable: &'a mut Vec<DynamicCollection<Prov>>,
  ctx: &'a Prov,
) {
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
}
