use crate::common::element::Element;
use crate::runtime::provenance::Tag;

pub fn collect_chosen_elements<'a, T: Tag, E: Element<T>>(all: &'a Vec<E>, chosen_ids: &Vec<usize>) -> Vec<&'a E> {
  all
    .iter()
    .enumerate()
    .filter(|(i, _)| chosen_ids.contains(i))
    .map(|(_, e)| e.clone())
    .collect::<Vec<_>>()
}
