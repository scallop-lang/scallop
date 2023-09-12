use pyo3::prelude::*;
use scallop_core::integrate::*;

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
    let mut args = vec![AttributeArgument::string(self.path.into())];
    if let Some(d) = self.deliminator {
      args.push(AttributeArgument::named_string("deliminator", d));
    }
    args.push(AttributeArgument::named_bool("has_header", self.has_header));
    args.push(AttributeArgument::named_bool("has_probability", self.has_probability));
    if let Some(keys) = self.keys {
      args.push(AttributeArgument::named_list("keys", keys.iter().cloned().map(AttributeValue::string).collect()));
    }
    if let Some(fields) = self.fields {
      args.push(AttributeArgument::named_list("fields", fields.iter().cloned().map(AttributeValue::string).collect()));
    }

    // Get attribute
    Attribute { name: "file".to_string(), args }
  }
}
