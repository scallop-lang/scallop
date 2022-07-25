use std::fs;
use std::path::PathBuf;

pub trait Source: std::fmt::Debug + 'static {
  fn content(&self) -> &str;

  fn file_path(&self) -> Option<PathBuf> {
    None
  }

  fn absolute_file_path(&self) -> Option<PathBuf> {
    self.file_path().map(|p| fs::canonicalize(p).unwrap())
  }

  fn resolve_import_file_path(&self, import_file_name: &str) -> PathBuf {
    if let Some(curr_file) = self.file_path() {
      let other_file_name = PathBuf::from(import_file_name);
      curr_file.parent().unwrap().join(other_file_name)
    } else {
      PathBuf::from(import_file_name.to_string())
    }
  }

  fn name(&self) -> Option<&str>;

  fn line(&self, line_num: usize) -> &str;

  fn line_name(&self, line_num: usize) -> String;

  fn num_rows(&self) -> usize;

  fn row_offset_length(&self, row: usize) -> (usize, usize);
}
