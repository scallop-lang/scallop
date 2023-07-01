use std::path::*;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum InputFile {
  Csv {
    file_path: PathBuf,
    deliminator: u8,
    has_header: bool,
    has_probability: bool,
    keys: Option<Vec<String>>,
    fields: Option<Vec<String>>,
  },
}

impl InputFile {
  pub fn csv(file_path: PathBuf) -> Self {
    Self::Csv {
      file_path,
      deliminator: b',',
      has_header: false,
      has_probability: false,
      keys: None,
      fields: None,
    }
  }

  pub fn csv_with_options(
    file_path: PathBuf,
    deliminator: Option<u8>,
    has_header: Option<bool>,
    has_probability: Option<bool>,
    keys: Option<Vec<String>>,
    fields: Option<Vec<String>>,
  ) -> Self {
    Self::Csv {
      file_path,
      deliminator: deliminator.unwrap_or(b','),
      has_header: has_header.unwrap_or(false) || keys.is_some() || fields.is_some(),
      has_probability: has_probability.unwrap_or(false),
      keys,
      fields,
    }
  }

  pub fn file_path(&self) -> &PathBuf {
    match self {
      Self::Csv { file_path, .. } => file_path,
    }
  }
}
