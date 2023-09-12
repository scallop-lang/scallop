use crate::compiler::front::*;

#[derive(Clone, Debug)]
pub struct DesugarArgTypeAdornment {
  new_items: Vec<Item>,
}

impl DesugarArgTypeAdornment {
  pub fn new() -> Self {
    Self { new_items: Vec::new() }
  }

  pub fn generate_demand_attribute(rel_type: &RelationType) -> Attribute {
    let pattern = rel_type.demand_pattern();
    Attribute::new(
      Identifier::new("demand".to_string()),
      vec![AttributeArg::Pos(AttributeValue::string(pattern))],
    )
  }

  pub fn retain_relation(&mut self, rel_type: &RelationType, existing_attrs: &Vec<Attribute>) -> bool {
    if rel_type.has_adornment() {
      let demand_attr = Self::generate_demand_attribute(rel_type);
      let attrs: Vec<_> = existing_attrs.iter().cloned().chain(std::iter::once(demand_attr)).collect();
      let item = Item::TypeDecl(TypeDecl::Relation(RelationTypeDecl::new(attrs, vec![rel_type.clone()])));
      self.new_items.push(item);
      false
    } else {
      true
    }
  }
}

impl NodeVisitor<RelationTypeDecl> for DesugarArgTypeAdornment {
  fn visit_mut(&mut self, relation_type_decl: &mut RelationTypeDecl) {
    if relation_type_decl.attrs().find("demand").is_none() {
      if relation_type_decl.rel_types().len() > 1 {
        let attributes = relation_type_decl.attrs().clone();
        relation_type_decl
          .rel_types_mut()
          .retain(|rel_type| self.retain_relation(rel_type, &attributes));
      } else {
        let rela_ty = relation_type_decl.get_rel_type(0).expect("Should not happen");
        if rela_ty.has_adornment() {
          let demand_attr = Self::generate_demand_attribute(rela_ty);
          relation_type_decl.attrs_mut().push(demand_attr);
        }
      }
    }
  }
}
