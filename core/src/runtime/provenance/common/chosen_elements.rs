use crate::common::element::Element;
use crate::runtime::provenance::*;

pub fn collect_chosen_elements<'a, Prov: Provenance, E: Element<Prov>>(
  all: &'a Vec<E>,
  chosen_ids: &Vec<usize>,
) -> Vec<&'a E> {
  all
    .iter()
    .enumerate()
    .filter(|(i, _)| chosen_ids.contains(i))
    .map(|(_, e)| e.clone())
    .collect::<Vec<_>>()
}
