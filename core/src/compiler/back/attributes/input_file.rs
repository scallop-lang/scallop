use crate::common::input_file::InputFile;

use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct InputFileAttribute {
  pub input_file: InputFile,
}

impl InputFileAttribute {
  pub fn new(input_file: InputFile) -> Self {
    Self { input_file }
  }
}

impl AttributeTrait for InputFileAttribute {
  fn name(&self) -> String {
    "file".to_string()
  }

  fn args(&self) -> Vec<String> {
    match &self.input_file {
      InputFile::Csv {
        file_path,
        deliminator,
        has_header,
        has_probability,
        keys,
        fields,
      } => {
        vec![
          "csv".to_string(),
          format!("file_path={:?}", file_path.display()),
          format!("deliminator={:?}", *deliminator as char),
          format!("has_header={}", has_header),
          format!("has_probability={}", has_probability),
          format!("keys={:?}", keys),
          format!("fields={:?}", fields),
        ]
      }
    }
  }
}
