use pyo3::prelude::*;
use scallop_core::integrate::Attribute;
use std::collections::*;

#[derive(FromPyObject)]
pub struct CSVFileOptions {
  pub path: String,
  pub deliminator: Option<String>,
  pub has_header: bool,
  pub has_probability: bool,
  pub keys: Option<Vec<String>>,
  pub fields: Option<Vec<String>>,
}

impl CSVFileOptions {
  pub fn new(path: String) -> Self {
    Self {
      path,
      deliminator: None,
      has_header: false,
      has_probability: false,
      keys: None,
      fields: None,
    }
  }
}

impl Into<Attribute> for CSVFileOptions {
  fn into(self) -> Attribute {
    let mut kw_args = HashMap::new();
    if let Some(d) = self.deliminator {
      kw_args.insert("deliminator".to_string(), d.into());
    }
    kw_args.insert("has_header".to_string(), self.has_header.into());
    kw_args.insert("has_probability".to_string(), self.has_probability.into());
    if let Some(keys) = self.keys {
      kw_args.insert("keys".to_string(), keys.into());
    }
    if let Some(fields) = self.fields {
      kw_args.insert("fields".to_string(), fields.into());
    }

    // Get attribute
    Attribute {
      name: "file".to_string(),
      positional_arguments: vec![self.path.into()],
      keyword_arguments: kw_args,
    }
  }
}
