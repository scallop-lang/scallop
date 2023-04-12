use pyo3::{Py, PyAny, Python};
use scallop_core::runtime::provenance;

/// The custom tag which holds an arbitrary python object.
#[derive(Clone, Debug)]
pub struct CustomTag(pub Py<PyAny>);

impl CustomTag {
  /// Create a new custom tag with a python object.
  pub fn new(tag: Py<PyAny>) -> Self {
    Self(tag)
  }
}

impl std::fmt::Display for CustomTag {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    std::fmt::Debug::fmt(&self, f)
  }
}

/// Custom tag is a tag
impl provenance::Tag for CustomTag {}

/// The custom provenance which is a wrapper of a python class
#[derive(Clone, Debug)]
pub struct CustomProvenance(pub Py<PyAny>);

impl provenance::Provenance for CustomProvenance {
  type Tag = CustomTag;

  type InputTag = Py<PyAny>;

  type OutputTag = Py<PyAny>;

  fn name() -> &'static str {
    "scallopy-custom"
  }

  /// Invoking the provenance's tagging function on the input tag
  fn tagging_fn(&self, i: Self::InputTag) -> Self::Tag {
    Python::with_gil(|py| {
      Self::Tag::new(self.0.call_method(py, "tagging_fn", (i,), None).unwrap())
    })
  }

  /// Invoking the provenance's recover function on an internal tag
  fn recover_fn(&self, t: &Self::Tag) -> Self::OutputTag {
    Python::with_gil(|py| {
      self
        .0
        .call_method(py, "recover_fn", (t.0.clone(),), None)
        .unwrap()
        .extract(py)
        .unwrap()
    })
  }

  /// Invoking the provenance's discard function on an internal tag
  fn discard(&self, t: &Self::Tag) -> bool {
    Python::with_gil(|py| {
      self
        .0
        .call_method(py, "discard", (t.0.clone(),), None)
        .unwrap()
        .extract(py)
        .unwrap()
    })
  }

  fn zero(&self) -> Self::Tag {
    Python::with_gil(|py| Self::Tag::new(self.0.call_method(py, "zero", (), None).unwrap().extract(py).unwrap()))
  }

  fn one(&self) -> Self::Tag {
    Python::with_gil(|py| Self::Tag::new(self.0.call_method(py, "one", (), None).unwrap().extract(py).unwrap()))
  }

  fn add(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    Python::with_gil(|py| {
      let input = (t1.0.clone(), t2.0.clone());
      Self::Tag::new(self.0.call_method(py, "add", input, None).unwrap().extract(py).unwrap())
    })
  }

  fn mult(&self, t1: &Self::Tag, t2: &Self::Tag) -> Self::Tag {
    Python::with_gil(|py| {
      let input = (t1.0.clone(), t2.0.clone());
      Self::Tag::new(
        self
          .0
          .call_method(py, "mult", input, None)
          .unwrap()
          .extract(py)
          .unwrap(),
      )
    })
  }

  fn negate(&self, t: &Self::Tag) -> Option<Self::Tag> {
    Python::with_gil(|py| {
      let input = (t.0.clone(),);
      Some(Self::Tag::new(
        self
          .0
          .call_method(py, "negate", input, None)
          .unwrap()
          .extract(py)
          .unwrap(),
      ))
    })
  }

  fn saturated(&self, t_old: &Self::Tag, t_new: &Self::Tag) -> bool {
    Python::with_gil(|py| {
      let input = (t_old.0.clone(), t_new.0.clone());
      self
        .0
        .call_method(py, "saturated", input, None)
        .unwrap()
        .extract(py)
        .unwrap()
    })
  }
}
