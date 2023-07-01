use crate::compiler::front::analyzers::*;
use crate::compiler::front::visitor_mut::*;
use crate::compiler::front::*;

#[derive(Debug)]
pub struct TransformAlgebraicDataType<'a> {
  analysis: &'a mut AlgebraicDataTypeAnalysis,
}

impl<'a> NodeVisitorMut for TransformAlgebraicDataType<'a> {
  fn visit_type_decl(&mut self, type_decl: &mut TypeDecl) {
    match &type_decl.node {
      TypeDeclNode::Algebraic(adt_decl) => {
        let location = adt_decl.location().clone();
        let name_identifier = adt_decl.name_identifier().clone();
        let new_decl = TypeDecl::alias(name_identifier.clone(), Type::entity()).with_location(location);
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
        let first_arg: Type = TypeNode::Named(variant_info.belongs_to_type.name().to_string()).into();
        let arg_types: Vec<ArgTypeBinding> = std::iter::once(first_arg)
          .chain(variant_info.args.iter().cloned())
          .map(|arg| ArgTypeBindingNode { name: None, ty: arg }.into())
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
            let constant = Constant::boolean(arg_is_entity);
            let attr_arg = AttributeValue::constant(constant);
            attr_arg
          })
          .collect();
        let adt_attr: Attribute = AttributeNode {
          name: Identifier::default_with_name("adt".to_string()),
          pos_args: vec![
            AttributeValue::constant(Constant::string(variant_name.clone())),
            is_entity,
          ],
          kw_args: vec![],
        }
        .into();

        // Generate another attribute `@hidden`
        let hidden_attr: Attribute = AttributeNode {
          name: Identifier::default_with_name("hidden".to_string()),
          pos_args: vec![],
          kw_args: vec![],
        }
        .into();

        // Generate a type declaration item
        Item::TypeDecl(
          TypeDeclNode::Relation(
            RelationTypeDeclNode {
              attrs: vec![adt_attr, hidden_attr],
              rel_types: vec![RelationTypeNode {
                name: rel_name,
                arg_types,
              }
              .into()],
            }
            .into(),
          )
          .into(),
        )
      })
      .collect();

    // Clear the variants
    self.analysis.adt_variants.clear();

    // Return the results
    result
  }

  pub fn retain(&self, item: &Item) -> bool {
    match item {
      Item::TypeDecl(td) => match td.node {
        TypeDeclNode::Algebraic(_) => false,
        _ => true,
      },
      _ => true,
    }
  }
}
