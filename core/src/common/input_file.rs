use std::path::*;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum InputFile {
  Csv {
    file_path: PathBuf,
    deliminator: u8,
    has_header: bool,
    has_probability: bool,
  },
  Txt(PathBuf),
}

impl InputFile {
  pub fn csv(file_path: PathBuf) -> Self {
    Self::Csv {
      file_path,
      deliminator: b',',
      has_header: false,
      has_probability: false,
    }
  }

  pub fn csv_with_options(
    file_path: PathBuf,
    deliminator: Option<u8>,
    has_header: Option<bool>,
    has_probability: Option<bool>,
  ) -> Self {
    Self::Csv {
      file_path,
      deliminator: deliminator.unwrap_or(b','),
      has_header: has_header.unwrap_or(false),
      has_probability: has_probability.unwrap_or(false),
    }
  }
}
