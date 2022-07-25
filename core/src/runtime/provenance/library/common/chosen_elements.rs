use crate::runtime::dynamic::DynamicElement;
use crate::runtime::provenance::Tag;

pub fn collect_chosen_elements<T: Tag>(
  all: &Vec<DynamicElement<T>>,
  chosen_ids: &Vec<usize>,
) -> Vec<DynamicElement<T>> {
  all
    .iter()
    .enumerate()
    .filter(|(i, _)| chosen_ids.contains(i))
    .map(|(_, e)| e.clone())
    .collect::<Vec<_>>()
}
