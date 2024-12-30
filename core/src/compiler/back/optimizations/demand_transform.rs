use std::collections::*;

use attributes::{DemandAttribute, MagicSetAttribute};

use crate::common::value_type::ValueType;

use super::super::*;

impl Program {
  pub fn demand_transform(&mut self) -> Result<(), BackCompileError> {
    // Compute adornments
    let adornments = collect_adornments(&self.relations).map_err(BackCompileError::from)?;
    if adornments.is_empty() {
      return Ok(());
    }

    // If there is non-empty adornment, we perform whole program transformation
    let new_rules = demand_transform_with_ctx(&self.rules, &adornments).map_err(BackCompileError::from)?;

    // If success, update rules
    self.rules = new_rules;

    // Update relations
    for (_, adornment) in adornments {
      self.relations.push(Relation {
        attributes: MagicSetAttribute.into(),
        predicate: adornment.demand_predicate.clone(),
        arg_types: adornment.demand_relation_types(),
      });
    }

    // Return ok
    Ok(())
  }
}

fn collect_adornments(relations: &Vec<Relation>) -> Result<HashMap<String, Adornment>, DemandTransformError> {
  let mut adornments = HashMap::new();
  for relation in relations {
    if let Some(demand_attr) = relation.attributes.get::<DemandAttribute>() {
      let p = Pattern::from_str(&demand_attr.pattern);
      let p = p.ok_or_else(|| DemandTransformError::InvalidPattern {
        pattern: demand_attr.pattern.clone(),
      })?;
      let a = Adornment::try_new(relation, p)?;
      adornments.insert(relation.predicate.clone(), a);
    }
  }
  Ok(adornments)
}

fn demand_transform_with_ctx(
  existing_rules: &Vec<Rule>,
  adornments: &HashMap<String, Adornment>,
) -> Result<Vec<Rule>, DemandTransformError> {
  let mut transformed_rules = vec![];
  let mut demand_rules = vec![];
  let mut other_rules = vec![];

  // Do full program demand transformation
  for rule in existing_rules {
    // Check is on-demand
    let maybe_adornment = adornments.get(rule.head_predicate());
    let to_check_rule = if let Some(adornment) = maybe_adornment {
      let transformed_rule = transform_on_demand_rule(rule, adornment);
      transformed_rules.push(transformed_rule);
      transformed_rules.last().unwrap()
    } else {
      other_rules.push(rule.clone());
      other_rules.last().unwrap()
    };

    // Check if we can create demand rule
    if contains_on_demand_predicate(to_check_rule, adornments) {
      let drs = generate_demand_rules(to_check_rule, adornments);
      demand_rules.extend(drs);
    }
  }

  // Return all the rules combined
  Ok(vec![transformed_rules, demand_rules, other_rules].concat())
}

fn transform_on_demand_rule(rule: &Rule, adornment: &Adornment) -> Rule {
  let mut new_rule = rule.clone();

  // Create demand atom
  let demand_atom = Atom {
    predicate: adornment.demand_predicate.clone(),
    args: adornment.pattern.get_bounded_args(&rule.head.get_atom().unwrap().args),
  };

  // Append it to the new rule
  new_rule.body.args.push(Literal::Atom(demand_atom));

  // Return new rule
  new_rule
}

fn contains_on_demand_predicate(rule: &Rule, adornments: &HashMap<String, Adornment>) -> bool {
  rule.body_literals().any(|l| match l {
    Literal::Atom(a) => adornments.contains_key(&a.predicate),
    Literal::NegAtom(a) => adornments.contains_key(&a.atom.predicate),
    _ => false,
  })
}

fn generate_demand_rules(rule: &Rule, adornments: &HashMap<String, Adornment>) -> Vec<Rule> {
  // First find all the atoms of demanded predicates
  let mut base = vec![];
  let mut to_ground = vec![];

  // Populate base/to_ground
  for lit in rule.body_literals() {
    match lit {
      Literal::Atom(a) => {
        if let Some(adornment) = adornments.get(&a.predicate) {
          to_ground.push(OnDemandAtom {
            atom: a.clone(),
            demand: adornment.clone(),
            vars: adornment.pattern.get_bounded_vars(&a.args),
          });
        } else {
          base.push(lit.clone());
        }
      }
      Literal::NegAtom(n) => {
        let a = &n.atom;
        if let Some(adornment) = adornments.get(&a.predicate) {
          to_ground.push(OnDemandAtom {
            atom: a.clone(),
            demand: adornment.clone(),
            vars: adornment.pattern.get_bounded_vars(&a.args),
          });
        } else {
          base.push(lit.clone());
        }
      }
      _ => {
        base.push(lit.clone());
      }
    }
  }

  // Invoke SIPS to find the information passing arcs.
  // In case a SIPS fails, no demand rule will be generated.
  let sips_gen = SIPSGenerator::MaxNumAtomsPerLayer;
  let maybe_arcs = generate_sips(&sips_gen, base, to_ground);
  if let Some(arcs) = maybe_arcs {
    // Iterate through all generated arcs
    let mut demand_rules = vec![];
    for arc in arcs {
      // For each arc and each of its rhs, generate a demand rule
      for to_ground in arc.rhs {
        let maybe_dr = generate_demand_rule(&arc.lhs, &to_ground.atom, &to_ground.demand);
        if let Some(dr) = maybe_dr {
          demand_rules.push(dr);
        }
      }
    }
    demand_rules
  } else {
    vec![]
  }
}

fn generate_demand_rule(base: &Vec<Literal>, goal: &Atom, adm: &Adornment) -> Option<Rule> {
  let base = remove_ungrounded_literals(base);
  if is_identity_demand_rule(&base, goal) {
    None
  } else {
    let rule = Rule {
      attributes: Attributes::new(),
      head: Head::atom(adm.demand_predicate.clone(), adm.pattern.get_bounded_args(&goal.args)),
      body: Conjunction { args: base },
    };
    Some(rule)
  }
}

fn generate_sips(sips_gen: &SIPSGenerator, base: Vec<Literal>, to_ground_atoms: Vec<OnDemandAtom>) -> Option<Vec<Arc>> {
  match sips_gen {
    SIPSGenerator::MaxNumAtomsPerLayer => generate_sips_with_max_num_atoms_per_layer(base, to_ground_atoms),
  }
}

fn generate_sips_with_max_num_atoms_per_layer(
  base: Vec<Literal>,
  to_ground_atoms: Vec<OnDemandAtom>,
) -> Option<Vec<Arc>> {
  // Prelude
  let mut arcs = vec![];
  let mut current_base = base;
  let mut current_to_ground_atoms = to_ground_atoms;

  // Fixpoint iteration
  loop {
    let grounded_vars = compute_grounded_variables(&current_base);

    // Compute the new things
    let mut new_to_ground_atoms = vec![];
    let mut new_grounded = vec![];
    for on_demand_atom in current_to_ground_atoms {
      if on_demand_atom.vars.is_subset(&grounded_vars) {
        new_grounded.push(on_demand_atom);
      } else {
        new_to_ground_atoms.push(on_demand_atom);
      }
    }

    // Setup arc
    arcs.push(Arc {
      lhs: current_base.clone(),
      rhs: new_grounded.clone(),
    });

    // Stopping condition
    if new_to_ground_atoms.is_empty() {
      // Success, since no more atom needs to be grounded
      break;
    } else if new_grounded.is_empty() {
      // Failure, since there are still atoms to be grounded,
      // but the iteration is not able to ground anything new
      return None;
    } else {
      // Need to continue, update the invariants
      current_base.extend(new_grounded.into_iter().map(|tg| Literal::Atom(tg.atom)));
      current_to_ground_atoms = new_to_ground_atoms;
    }
  }

  // Return the generated arcs in the end
  Some(arcs)
}

fn compute_grounded_variables(base: &Vec<Literal>) -> HashSet<Variable> {
  let mut grounded = base
    .iter()
    .flat_map(|lit| match lit {
      Literal::Atom(a) => a.variable_args().collect::<Vec<_>>(),
      _ => vec![],
    })
    .collect::<HashSet<_>>();

  // Fixpoint iteration
  loop {
    let mut new_grounded = grounded.clone();
    new_grounded.extend(base.iter().flat_map(|lit| {
      match lit {
        Literal::Assign(a) => {
          if a
            .variable_args()
            .into_iter()
            .collect::<HashSet<_>>()
            .is_subset(&grounded)
          {
            vec![&a.left]
          } else {
            vec![]
          }
        }
        _ => vec![],
      }
    }));

    // Reaches fixpoint, return
    if new_grounded == grounded {
      break grounded.into_iter().cloned().collect();
    } else {
      grounded = new_grounded;
    }
  }
}

fn remove_ungrounded_literals(base: &Vec<Literal>) -> Vec<Literal> {
  let grounded = compute_grounded_variables(base);
  base
    .iter()
    .filter(|lit| match lit {
      Literal::Assign(a) => a
        .variable_args()
        .into_iter()
        .cloned()
        .collect::<HashSet<_>>()
        .is_subset(&grounded),
      Literal::Constraint(c) => c
        .variable_args()
        .into_iter()
        .cloned()
        .collect::<HashSet<_>>()
        .is_subset(&grounded),
      _ => true,
    })
    .cloned()
    .collect()
}

fn is_identity_demand_rule(base: &Vec<Literal>, goal: &Atom) -> bool {
  if base.len() == 1 {
    if let Literal::Atom(a) = &base[0] {
      if a == goal {
        return true;
      }
    }
  }
  false
}

#[derive(Clone, Debug)]
enum Boundness {
  Bound,
  Free,
}

impl Boundness {
  fn is_bound(&self) -> bool {
    match self {
      Self::Bound => true,
      _ => false,
    }
  }

  fn from_char(c: char) -> Option<Self> {
    if c == 'b' {
      Some(Boundness::Bound)
    } else if c == 'f' {
      Some(Boundness::Free)
    } else {
      None
    }
  }

  fn to_char(&self) -> char {
    match self {
      Boundness::Bound => 'b',
      Boundness::Free => 'f',
    }
  }
}

#[derive(Clone, Debug)]
struct Pattern(Box<[Boundness]>);

impl Pattern {
  fn to_string(&self) -> String {
    self.0.iter().map(|b| Boundness::to_char(b)).collect()
  }

  fn len(&self) -> usize {
    self.0.len()
  }

  fn get_bounded_args(&self, args: &Vec<Term>) -> Vec<Term> {
    args
      .iter()
      .zip(self.0.iter())
      .filter_map(|(a, b)| if b.is_bound() { Some(a.clone()) } else { None })
      .collect()
  }

  fn get_bounded_vars(&self, args: &Vec<Term>) -> HashSet<Variable> {
    args
      .iter()
      .zip(self.0.iter())
      .filter_map(|(a, b)| {
        if b.is_bound() {
          match a {
            Term::Variable(v) => Some(v.clone()),
            _ => None,
          }
        } else {
          None
        }
      })
      .collect()
  }

  fn from_str(s: &str) -> Option<Self> {
    s.chars().map(Boundness::from_char).collect::<Option<_>>().map(Self)
  }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct Adornment {
  predicate: String,
  types: Vec<ValueType>,
  pattern: Pattern,
  demand_predicate: String,
}

impl Adornment {
  fn try_new(relation: &Relation, pattern: Pattern) -> Result<Self, DemandTransformError> {
    // Check arity
    if relation.arg_types.len() != pattern.len() {
      return Err(DemandTransformError::ArityMismatch {
        predicate: relation.predicate.clone(),
        pattern: pattern.to_string(),
        expected: relation.arg_types.len(),
        actual: pattern.len(),
      });
    }

    // Create demand predicate
    let demand_predicate = format!("d#{}#{}", relation.predicate.clone(), pattern.to_string());
    Ok(Self {
      predicate: relation.predicate.clone(),
      types: relation.arg_types.clone(),
      pattern,
      demand_predicate,
    })
  }

  fn demand_relation_types(&self) -> Vec<ValueType> {
    self
      .types
      .iter()
      .zip(self.pattern.0.iter())
      .filter_map(|(ty, b)| if b.is_bound() { Some(ty.clone()) } else { None })
      .collect()
  }
}

#[derive(Clone, Debug)]
pub enum DemandTransformError {
  InvalidPattern {
    pattern: String,
  },
  ArityMismatch {
    predicate: String,
    pattern: String,
    expected: usize,
    actual: usize,
  },
}

impl From<DemandTransformError> for BackCompileError {
  fn from(e: DemandTransformError) -> Self {
    Self::DemandTransformError(e)
  }
}

impl std::fmt::Display for DemandTransformError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::InvalidPattern { pattern } => f.write_fmt(format_args!("Invalid pattern `{}`", pattern)),
      Self::ArityMismatch {
        predicate,
        pattern,
        expected,
        actual,
      } => f.write_fmt(format_args!(
        "Arity mismatch of demand pattern `{}` for `{}`. Expected {}, found {}.",
        pattern, predicate, expected, actual
      )),
    }
  }
}

#[derive(Clone, Debug)]
enum SIPSGenerator {
  MaxNumAtomsPerLayer,
}

#[derive(Clone, Debug)]
struct Arc {
  lhs: Vec<Literal>,
  rhs: Vec<OnDemandAtom>,
}

#[derive(Clone, Debug)]
struct OnDemandAtom {
  atom: Atom,
  demand: Adornment,
  vars: HashSet<Variable>,
}
