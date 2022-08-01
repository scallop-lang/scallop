use std::path::PathBuf;

use super::*;

#[derive(Clone, Debug)]
pub struct FileSource {
  pub file_name: String,
  pub file_content: String,
  pub line_offset_len: Vec<(usize, usize)>,
}

impl FileSource {
  pub fn new(file_path: &PathBuf) -> Result<Self, SourceError> {
    // 1. Store the file name
    let file_name = String::from(file_path.to_str().unwrap());

    // 2. Load the file content
    let file_content = std::fs::read_to_string(file_path).map_err(|e| SourceError::CannotOpenFile {
      file_name: file_path.clone(),
      std_io_error: format!("{}", e),
    })?;

    // 3. Compute the line/row offsets
    let line_offset_len = collect_line_offset_length(&file_content);

    // 4. Return!
    Ok(Self {
      file_name,
      file_content,
      line_offset_len,
    })
  }
}

impl Source for FileSource {
  fn content(&self) -> &str {
    &self.file_content
  }

  fn file_path(&self) -> Option<PathBuf> {
    Some(PathBuf::from(self.file_name.clone()))
  }

  fn name(&self) -> Option<&str> {
    Some(&self.file_name)
  }

  fn line(&self, line_num: usize) -> &str {
    let (off, len) = &self.line_offset_len[line_num];
    &self.file_content[*off..off + len]
  }

  fn line_name(&self, line_num: usize) -> String {
    format!("{}", line_num + 1)
  }

  fn num_rows(&self) -> usize {
    self.line_offset_len.len()
  }

  fn row_offset_length(&self, row: usize) -> (usize, usize) {
    self.line_offset_len[row]
  }
}
