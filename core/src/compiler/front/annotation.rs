use crate::utils::IdAllocator;

use super::*;

#[derive(Clone, Debug)]
pub struct NodeIdAnnotator {
  pub id_allocator: IdAllocator,
}

impl NodeIdAnnotator {
  pub fn new() -> Self {
    Self {
      id_allocator: IdAllocator::default(),
    }
  }

  pub fn annotate_items(&mut self, items: &mut Vec<Item>) {
    items.walk(self);
  }

  pub fn annotate_item(&mut self, item: &mut Item) {
    item.walk(self);
  }
}

impl NodeVisitor<NodeLocation> for NodeIdAnnotator {
  fn visit_mut(&mut self, loc: &mut NodeLocation) {
    if loc.id.is_none() {
      loc.id = Some(self.id_allocator.alloc());
    }
  }
}

#[derive(Clone, Debug)]
pub struct SourceIdAnnotator {
  pub source_id: usize,
}

impl SourceIdAnnotator {
  pub fn new(source_id: usize) -> Self {
    Self { source_id }
  }
}

impl NodeVisitor<NodeLocation> for SourceIdAnnotator {
  fn visit_mut(&mut self, loc: &mut NodeLocation) {
    loc.source_id = self.source_id;
  }
}

#[derive(Clone, Debug)]
pub struct LocationSpanAnnotator {
  pub row_offset_length: Vec<(usize, usize)>,
}

impl LocationSpanAnnotator {
  pub fn new<S: Source>(source: &S) -> Self {
    Self {
      row_offset_length: (0..source.num_rows()).map(|i| source.row_offset_length(i)).collect(),
    }
  }

  pub fn row_col_of_offset(&self, offset: &usize) -> CharLocation {
    let num_rows = self.row_offset_length.len();
    for (i, (curr_offset, _)) in self.row_offset_length.iter().enumerate() {
      if curr_offset <= offset && (i == num_rows - 1 || offset < &self.row_offset_length[i + 1].0) {
        return CharLocation {
          row: i,
          col: offset - curr_offset,
        };
      }
    }
    CharLocation { row: 0, col: 0 }
  }
}

impl NodeVisitor<NodeLocation> for LocationSpanAnnotator {
  fn visit_mut(&mut self, loc: &mut NodeLocation) {
    let offset_start = &loc.offset_span.start;
    let offset_end = &loc.offset_span.end;
    let start = self.row_col_of_offset(offset_start);
    let end = self.row_col_of_offset(offset_end);
    loc.loc_span = Some(Span { start, end });
  }
}
