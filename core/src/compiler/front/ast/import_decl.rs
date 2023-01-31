use super::*;

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct ImportFileNode {
  pub file_path: String,
}

pub type ImportFile = AstNode<ImportFileNode>;

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)]
pub struct ImportDeclNode {
  pub attrs: Attributes,
  pub import_file: ImportFile,
}

pub type ImportDecl = AstNode<ImportDeclNode>;

impl ImportDecl {
  pub fn attributes(&self) -> &Attributes {
    &self.node.attrs
  }

  pub fn attributes_mut(&mut self) -> &mut Attributes {
    &mut self.node.attrs
  }

  pub fn input_file(&self) -> &String {
    &self.node.import_file.node.file_path
  }
}
