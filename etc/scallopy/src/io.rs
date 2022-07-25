use pyo3::prelude::*;
use scallop_core::integrate::Attribute;
use std::collections::*;

#[derive(FromPyObject)]
pub struct CSVFileOptions {
  pub path: String,
  pub deliminator: Option<String>,
  pub has_header: bool,
  pub has_probability: bool,
}

impl CSVFileOptions {
  pub fn new(path: String) -> Self {
    Self {
      path,
      deliminator: None,
      has_header: false,
      has_probability: false,
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

    // Get attribute
    Attribute {
      name: "file".to_string(),
      positional_arguments: vec![self.path.into()],
      keyword_arguments: kw_args,
    }
  }
}
