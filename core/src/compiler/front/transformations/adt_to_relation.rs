use crate::compiler::front::analyzers::*;
use crate::compiler::front::*;

#[derive(Debug)]
pub struct TransformAlgebraicDataType<'a> {
  analysis: &'a mut AlgebraicDataTypeAnalysis,
}

impl<'a> NodeVisitor<TypeDecl> for TransformAlgebraicDataType<'a> {
  fn visit_mut(&mut self, type_decl: &mut TypeDecl) {
    match type_decl {
      TypeDecl::Algebraic(adt_decl) => {
        let location = adt_decl.location().clone();
        let name = adt_decl.name().clone();
        let new_decl = TypeDecl::alias(AliasTypeDecl::new_with_loc(vec![], name, Type::entity(), location));
        *type_decl = new_decl;
      }
      _ => {}
    }
  }
}

impl<'a> TransformAlgebraicDataType<'a> {
  pub fn new(analysis: &'a mut AlgebraicDataTypeAnalysis) -> Self {
    Self { analysis }
  }

  pub fn generate_items(self) -> Vec<Item> {
    let result = self
      .analysis
      .adt_variants
      .iter()
      .map(|(variant_name, variant_info)| {
        let rel_name = variant_info
          .name
          .clone_without_location_id()
          .map(|n| format!("adt#{n}"));

        // Generate the args including the first ID type
        let first_arg = Type::named(variant_info.belongs_to_type.name().to_string());
        let arg_types: Vec<ArgTypeBinding> = std::iter::once(first_arg)
          .chain(variant_info.args.iter().cloned())
          .map(|arg| ArgTypeBinding::new(None, None, arg))
          .collect();

        // Generate an attribute `@adt("VARIANT_NAME", [IS_ARG_0_ENTITY, ...])`
        let is_entity: AttributeValue = variant_info
          .args
          .iter()
          .map(|arg| {
            let arg_is_entity = if let Some(name) = arg.get_name() {
              self.analysis.adt_types.contains(name)
            } else {
              false
            };
            let constant = Constant::boolean(BoolLiteral::new(arg_is_entity));
            let attr_arg = AttributeValue::constant(constant);
            attr_arg
          })
          .collect();
        let adt_attr = Attribute::new(
          Identifier::new("adt".to_string()),
          vec![AttributeValue::string(variant_name.clone()).into(), is_entity.into()],
        );

        // Generate another attribute `@hidden`
        let hidden_attr = Attribute::new(Identifier::new("hidden".to_string()), vec![]);

        // Generate a type declaration item
        Item::TypeDecl(TypeDecl::Relation(RelationTypeDecl::new(
          vec![adt_attr, hidden_attr],
          None,
          vec![RelationType::new(rel_name, arg_types)],
        )))
      })
      .collect();

    // Clear the variants
    self.analysis.adt_variants.clear();

    // Return the results
    result
  }

  pub fn retain(&self, item: &Item) -> bool {
    match item {
      Item::TypeDecl(TypeDecl::Algebraic(_)) => false,
      _ => true,
    }
  }
}
