use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub enum TypeDecl {
  Subtype(SubtypeDecl),
  Alias(AliasTypeDecl),
  Relation(RelationTypeDecl),
  Enumerate(EnumTypeDecl),
  Algebraic(AlgebraicDataTypeDecl),
  Function(FunctionTypeDecl),
}

impl TypeDecl {
  pub fn attrs(&self) -> &Attributes {
    match self {
      TypeDecl::Subtype(s) => s.attrs(),
      TypeDecl::Alias(a) => a.attrs(),
      TypeDecl::Relation(r) => r.attrs(),
      TypeDecl::Enumerate(e) => e.attrs(),
      TypeDecl::Algebraic(a) => a.attrs(),
      TypeDecl::Function(a) => a.attrs(),
    }
  }

  pub fn attrs_mut(&mut self) -> &mut Attributes {
    match self {
      TypeDecl::Subtype(s) => s.attrs_mut(),
      TypeDecl::Alias(a) => a.attrs_mut(),
      TypeDecl::Relation(r) => r.attrs_mut(),
      TypeDecl::Enumerate(e) => e.attrs_mut(),
      TypeDecl::Algebraic(a) => a.attrs_mut(),
      TypeDecl::Function(a) => a.attrs_mut(),
    }
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _SubtypeDecl {
  pub attrs: Attributes,
  pub name: Identifier,
  pub subtype_of: Type,
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _AliasTypeDecl {
  pub attrs: Attributes,
  pub name: Identifier,
  pub alias_of: Type,
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub enum _ArgTypeBindingAdornment {
  Bound,
  Free,
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _ArgTypeBinding {
  pub adornment: Option<ArgTypeBindingAdornment>,
  pub name: Option<Identifier>,
  pub ty: Type,
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _RelationType {
  pub name: Identifier,
  pub arg_bindings: Vec<ArgTypeBinding>,
}

impl Into<Vec<Item>> for RelationType {
  fn into(self) -> Vec<Item> {
    vec![Item::TypeDecl(TypeDecl::Relation(RelationTypeDecl::new(
      Attributes::new(),
      None,
      vec![self],
    )))]
  }
}

impl RelationType {
  pub fn has_adornment(&self) -> bool {
    self.iter_arg_bindings().any(|arg_ty| arg_ty.adornment().is_some())
  }

  pub fn demand_pattern(&self) -> String {
    self
      .iter_arg_bindings()
      .map(|arg| match arg.adornment() {
        Some(anno) if anno.is_bound() => "b",
        _ => "f",
      })
      .collect::<Vec<_>>()
      .join("")
  }

  pub fn predicate_name(&self) -> &String {
    self.name().name()
  }

  pub fn has_arg_name(&self) -> bool {
    self.iter_arg_bindings().any(|a| a.name().is_some())
  }

  pub fn iter_arg_types(&self) -> impl Iterator<Item = &Type> {
    self.iter_arg_bindings().map(|arg| arg.ty())
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _Extern;

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _RelationTypeDecl {
  pub attrs: Attributes,
  pub ext: Option<Extern>,
  pub rel_types: Vec<RelationType>,
}

impl RelationTypeDecl {
  pub fn get_rel_type(&self, i: usize) -> Option<&RelationType> {
    self.rel_types().get(i)
  }

  pub fn predicates(&self) -> impl Iterator<Item = &String> {
    self.iter_rel_types().map(|rel_type| rel_type.name().name())
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
#[doc(hidden)]
pub struct _EnumTypeDecl {
  pub attrs: Attributes,
  pub name: Identifier,
  pub members: Vec<EnumTypeMember>,
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
pub struct _EnumTypeMember {
  pub name: Identifier,
  pub assigned_num: Option<Constant>,
}

impl EnumTypeMember {
  pub fn member_name(&self) -> &String {
    self.name().name()
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
pub struct _AlgebraicDataTypeDecl {
  pub attrs: Attributes,
  pub name: Identifier,
  pub variants: Vec<AlgebraicDataTypeVariant>,
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
pub struct _AlgebraicDataTypeVariant {
  pub constructor: Identifier,
  pub args: Vec<Type>,
}

impl AlgebraicDataTypeVariant {
  pub fn constructor_name(&self) -> &str {
    self.constructor().name()
  }
}

#[derive(Clone, Debug, PartialEq, Serialize, AstNode)]
pub struct _FunctionTypeDecl {
  pub attrs: Attributes,
  pub func_name: Identifier,
  pub args: Vec<ArgTypeBinding>,
  pub ret_ty: Type,
}
