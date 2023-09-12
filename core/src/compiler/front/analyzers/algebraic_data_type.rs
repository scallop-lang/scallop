use std::collections::*;

use crate::compiler::front::*;

#[derive(Clone, Debug)]
pub struct AlgebraicDataTypeAnalysis {
  pub errors: Vec<ADTError>,
  pub adt_variants: HashMap<String, VariantInfo>,
  pub adt_types: HashSet<String>,
}

#[derive(Clone, Debug)]
pub struct VariantInfo {
  pub belongs_to_type: Identifier,
  pub name: Identifier,
  pub location: NodeLocation,
  pub args: Vec<Type>,
}

impl AlgebraicDataTypeAnalysis {
  pub fn new() -> Self {
    Self {
      errors: Vec::new(),
      adt_variants: HashMap::new(),
      adt_types: HashSet::new(),
    }
  }
}

impl NodeVisitor<AlgebraicDataTypeDecl> for AlgebraicDataTypeAnalysis {
  fn visit(&mut self, decl: &AlgebraicDataTypeDecl) {
    // Add the type to the set of adt types
    self.adt_types.insert(decl.name().to_string());

    // And then declare all the constant types
    let mut visited_names: HashMap<&str, &NodeLocation> = HashMap::new();
    for variant in decl.iter_variants() {
      // First check if the variant has already being declared
      if let Some(loc) = visited_names.get(variant.constructor_name()) {
        self.errors.push(ADTError::DuplicateADTVariant {
          constructor: variant.constructor_name().to_string(),
          first_declared: (*loc).clone(),
          duplicated: variant.location().clone(),
        });
      } else {
        visited_names.insert(variant.constructor_name(), variant.location());
      }

      // Then check if the variant has occurred previously
      if let Some(info) = self.adt_variants.get(variant.constructor_name()) {
        self.errors.push(ADTError::DuplicateADTVariant {
          constructor: variant.constructor_name().to_string(),
          first_declared: info.location.clone(),
          duplicated: variant.location().clone(),
        });
      }

      // If everything is well, store the ADT variant
      let info = VariantInfo {
        belongs_to_type: decl.name().clone(),
        name: variant.constructor().clone(),
        location: variant.location().clone(),
        args: variant.args().clone(),
      };
      self.adt_variants.insert(variant.constructor_name().to_string(), info);
    }
  }
}

#[derive(Clone, Debug)]
pub enum ADTError {
  DuplicateADTVariant {
    constructor: String,
    first_declared: NodeLocation,
    duplicated: NodeLocation,
  },
}

impl FrontCompileErrorTrait for ADTError {
  fn error_type(&self) -> FrontCompileErrorType {
    FrontCompileErrorType::Error
  }

  fn report(&self, src: &Sources) -> String {
    match self {
      Self::DuplicateADTVariant {
        constructor,
        first_declared,
        duplicated,
      } => {
        format!(
          "duplicated Algebraic Data Type variant `{}`. It is first declared here:\n{}\nwhile we find a duplicated declaration here:\n{}",
          constructor, first_declared.report(src), duplicated.report(src)
        )
      }
    }
  }
}
