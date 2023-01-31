use crate::common::input_file::InputFile;

#[derive(Clone, Debug, PartialEq)]
pub struct Attributes {
  pub attrs: Vec<Attribute>,
}

impl Attributes {
  pub fn new() -> Self {
    Self { attrs: vec![] }
  }

  pub fn singleton<A>(attr: A) -> Self
  where
    A: Into<Attribute>,
  {
    Self {
      attrs: vec![attr.into()],
    }
  }

  pub fn add_attribute(&mut self, attr: Attribute) {
    self.attrs.push(attr)
  }

  pub fn aggregate_body_attr(&self) -> Option<&AggregateBodyAttribute> {
    for attr in &self.attrs {
      match attr {
        Attribute::AggregateBody(a) => return Some(a),
        _ => {}
      }
    }
    None
  }

  pub fn aggregate_group_by_attr(&self) -> Option<&AggregateGroupByAttribute> {
    for attr in &self.attrs {
      match attr {
        Attribute::AggregateGroupBy(a) => return Some(a),
        _ => {}
      }
    }
    None
  }

  pub fn demand_attr(&self) -> Option<&DemandAttribute> {
    for attr in &self.attrs {
      match attr {
        Attribute::Demand(d) => return Some(d),
        _ => {}
      }
    }
    None
  }

  pub fn magic_set_attr(&self) -> Option<&MagicSetAttribute> {
    for attr in &self.attrs {
      match attr {
        Attribute::MagicSet(m) => return Some(m),
        _ => {}
      }
    }
    None
  }

  pub fn input_file_attr(&self) -> Option<&InputFileAttribute> {
    for attr in &self.attrs {
      match attr {
        Attribute::InputFile(d) => return Some(d),
        _ => {}
      }
    }
    None
  }
}

impl<I> From<I> for Attributes
where
  I: IntoIterator<Item = Attribute>,
{
  fn from(i: I) -> Self {
    Self {
      attrs: i.into_iter().collect(),
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Attribute {
  AggregateBody(AggregateBodyAttribute),
  AggregateGroupBy(AggregateGroupByAttribute),
  Demand(DemandAttribute),
  MagicSet(MagicSetAttribute),
  InputFile(InputFileAttribute),
}

impl Attribute {
  pub fn aggregate_body(num_group_by_vars: usize, num_arg_vars: usize, num_key_vars: usize) -> Self {
    Self::AggregateBody(AggregateBodyAttribute {
      num_group_by_vars,
      num_arg_vars,
      num_key_vars,
    })
  }

  pub fn aggregate_group_by(num_join_vars: usize, num_other_vars: usize) -> Self {
    Self::AggregateGroupBy(AggregateGroupByAttribute {
      num_join_group_by_vars: num_join_vars,
      num_other_group_by_vars: num_other_vars,
    })
  }

  pub fn magic_set() -> Self {
    Self::MagicSet(MagicSetAttribute)
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AggregateBodyAttribute {
  pub num_group_by_vars: usize,
  pub num_arg_vars: usize,
  pub num_key_vars: usize,
}

impl AggregateBodyAttribute {
  pub fn new(num_group_by_vars: usize, num_arg_vars: usize, num_key_vars: usize) -> Self {
    Self {
      num_group_by_vars,
      num_arg_vars,
      num_key_vars,
    }
  }
}

impl Into<Attribute> for AggregateBodyAttribute {
  fn into(self) -> Attribute {
    Attribute::AggregateBody(self)
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AggregateGroupByAttribute {
  pub num_join_group_by_vars: usize,
  pub num_other_group_by_vars: usize,
}

impl AggregateGroupByAttribute {
  pub fn new(num_join_group_by_vars: usize, num_other_group_by_vars: usize) -> Self {
    Self {
      num_join_group_by_vars,
      num_other_group_by_vars,
    }
  }
}

impl Into<Attribute> for AggregateGroupByAttribute {
  fn into(self) -> Attribute {
    Attribute::AggregateGroupBy(self)
  }
}

/// Demand attributes to the relations which are on-demand relations
#[derive(Clone, Debug, PartialEq)]
pub struct DemandAttribute {
  pub pattern: String,
}

/// Magic-Set attributes to the helper relations storing the demanded-tuples for on-demand relations.
/// These relations are called "magic-sets".
#[derive(Clone, Debug, PartialEq)]
pub struct MagicSetAttribute;

#[derive(Clone, Debug, PartialEq)]
pub struct InputFileAttribute {
  pub input_file: InputFile,
}
