use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _ImportDecl {
  pub attrs: Attributes,
  pub import_file: StringLiteral,
}

impl ImportDecl {
  pub fn import_file_path(&self) -> &String {
    &self.import_file().string()
  }
}
