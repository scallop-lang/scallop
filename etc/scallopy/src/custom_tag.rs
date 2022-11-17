use pyo3::{Py, PyAny, Python};
// use scallop_core::runtime::dynamic;
use scallop_core::runtime::provenance;

#[derive(Clone, Debug)]
pub struct CustomTag(pub Py<PyAny>);

impl CustomTag {
  pub fn new(tag: Py<PyAny>) -> Self {
    Self(tag)
  }
}

impl std::fmt::Display for CustomTag {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    std::fmt::Debug::fmt(&self, f)
  }
}

impl provenance::Tag for CustomTag {}

#[derive(Clone, Debug)]
pub struct CustomTagContext(pub Py<PyAny>);

impl provenance::Provenance for CustomTagContext {
  type Tag = CustomTag;

  type InputTag = Py<PyAny>;

  type OutputTag = Py<PyAny>;

  fn name() -> &'static str {
    "scallopy-custom"
  }

  fn tagging_fn(&mut self, i: Self::InputTag) -> Self::Tag {
    Python::with_gil(|py| {
      let result = self.0.call_method(py, "tagging_fn", (i,), None).unwrap();
      Self::Tag::new(result)
    })
  }

  fn tagging_disjunction_fn(&mut self, i: Vec<Self::InputTag>) -> Vec<Self::Tag> {
    Python::with_gil(|py| {
      let result: Vec<Py<PyAny>> = self
        .0
        .call_method(py, "tagging_disjunction_fn", (i,), None)
        .unwrap()
        .extract(py)
        .unwrap();
      result
        .into_iter()
        .map(|t| {
          let t: Py<PyAny> = t.into();
          Self::Tag::new(t)
        })
        .collect()
    })
  }

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
