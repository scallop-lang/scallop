use pyo3::types::*;
use pyo3::*;

use scallop_core::compiler::front::attribute::*;
use scallop_core::compiler::front::*;

use crate::foreign_predicate::PythonForeignPredicate;

#[derive(Clone)]
pub struct PythonForeignAttribute {
  py_attr: PyObject,
  name: String,
}

impl PythonForeignAttribute {
  pub fn new(py_attr: PyObject) -> Self {
    let name = Python::with_gil(|py| {
      py_attr
        .getattr(py, "name")
        .expect("Cannot get foreign predicate name")
        .extract(py)
        .expect("Foreign predicate name cannot be extracted into String")
    });

    Self { py_attr, name }
  }

  pub fn process_action(&self, py: Python, result: Py<PyAny>) -> AttributeAction {
    let name: String = result.getattr(py, "name").unwrap().extract(py).unwrap();
    match name.as_str() {
      "multiple" => {
        let py_actions: Vec<Py<PyAny>> = result.getattr(py, "actions").unwrap().extract(py).unwrap();
        let actions = py_actions
          .into_iter()
          .map(|py_action| self.process_action(py.clone(), py_action))
          .collect();
        AttributeAction::Multiple(actions)
      }
      "remove_item" => AttributeAction::RemoveItem,
      "no_action" => AttributeAction::Nothing,
      "error" => {
        let msg: String = result.getattr(py, "msg").unwrap().extract(py).unwrap();
        AttributeAction::Error(msg)
      }
      "register_foreign_predicate" => {
        let py_fp: Py<PyAny> = result.getattr(py, "foreign_predicate").unwrap().extract(py).unwrap();
        let fp = PythonForeignPredicate::new(py_fp);
        AttributeAction::Context(Box::new(move |ctx| {
          ctx
            .register_foreign_predicate(fp)
            .expect("Cannot register foreign predicate");
        }))
      }
      n => panic!("Unknown action `{}`", n),
    }
  }
}

impl AttributeProcessor for PythonForeignAttribute {
  fn name(&self) -> String {
    self.name.clone()
  }

  fn apply(&self, item: &ast::Item, attr: &ast::Attribute) -> Result<AttributeAction, AttributeError> {
    Python::with_gil(|py| {
      let item_py = pythonize::pythonize(py, item).unwrap();
      let attr_py = pythonize::pythonize(py, attr).unwrap();
      let args = PyTuple::new(py, vec![item_py, attr_py]);
      let result = self.py_attr.call_method(py, "apply", args, None).unwrap();
      Ok(self.process_action(py, result))
    })
  }
}
