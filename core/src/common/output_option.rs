use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd)]
pub enum OutputOption {
  /// Hidden output means that the relation would not be returned or printed by default
  Hidden,

  /// Default output means that the relation is either printed or returned
  Default,

  /// File output means that we will output the thing into a file
  File(OutputFile),
}

impl Default for OutputOption {
  fn default() -> Self {
    Self::Default
  }
}

impl OutputOption {
  pub fn is_hidden(&self) -> bool {
    match self {
      Self::Hidden => true,
      _ => false,
    }
  }

  pub fn is_default(&self) -> bool {
    match self {
      Self::Default => true,
      _ => false,
    }
  }

  pub fn is_not_hidden(&self) -> bool {
    !self.is_hidden()
  }
}

impl std::fmt::Display for OutputOption {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Hidden => f.write_str("hidden"),
      Self::Default => f.write_str("default"),
      Self::File(o) => std::fmt::Display::fmt(o, f),
    }
  }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd)]
pub enum OutputFile {
  CSV(OutputCSVFile),
}

impl std::fmt::Display for OutputFile {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::CSV(c) => std::fmt::Display::fmt(c, f),
    }
  }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct OutputCSVFile {
  pub file_path: PathBuf,
  pub deliminator: u8,
}

impl OutputCSVFile {
  pub fn new(file_path: PathBuf) -> Self {
    Self {
      file_path,
      deliminator: b',',
    }
  }

  pub fn new_with_options(file_path: PathBuf, deliminator: Option<u8>) -> Self {
    Self {
      file_path,
      deliminator: deliminator.unwrap_or(b','),
    }
  }
}

impl std::fmt::Display for OutputCSVFile {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_fmt(format_args!(
      "CSV(\"{:?}\", deliminator='{}')",
      self.file_path, self.deliminator as char
    ))
  }
}
