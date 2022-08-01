use std::collections::*;
use std::path::*;

use super::super::*;
use crate::common::input_file::InputFile;

#[derive(Clone, Debug)]
pub struct InputFilesAnalysis {
  pub input_files: HashMap<String, InputFile>,
  pub errors: Vec<InputFilesError>,
}

impl InputFilesAnalysis {
  pub fn new() -> Self {
    Self {
      input_files: HashMap::new(),
      errors: Vec::new(),
    }
  }

  pub fn add_input_file(&mut self, relation: String, input_file: InputFile) -> Result<(), InputFilesError> {
    self.input_files.insert(relation, input_file);
    Ok(())
  }

  pub fn input_file(&self, relation: &String) -> Option<&InputFile> {
    self.input_files.get(relation)
  }

  pub fn process_deliminator(&self, attr_arg: Option<&Constant>) -> Result<Option<u8>, InputFilesError> {
    match attr_arg {
      Some(v) => match &v.node {
        ConstantNode::String(s) => {
          if s.len() == 1 {
            let c = s.chars().next().unwrap();
            if c.is_ascii() {
              Ok(Some(c as u8))
            } else {
              Err(InputFilesError::DeliminatorNotASCII {
                loc: v.location().clone(),
              })
            }
          } else {
            Err(InputFilesError::DeliminatorNotSingleCharacter {
              loc: v.location().clone(),
            })
          }
        }
        _ => Err(InputFilesError::DeliminatorNotString {
          loc: v.location().clone(),
        }),
      },
      None => Ok(None),
    }
  }

  pub fn process_has_header(&self, attr_arg: Option<&Constant>) -> Result<Option<bool>, InputFilesError> {
    match attr_arg {
      Some(v) => match &v.node {
        ConstantNode::Boolean(b) => Ok(Some(*b)),
        _ => Err(InputFilesError::HasHeaderNotBoolean {
          loc: v.location().clone(),
        }),
      },
      None => Ok(None),
    }
  }

  pub fn process_has_probability(&self, attr_arg: Option<&Constant>) -> Result<Option<bool>, InputFilesError> {
    match attr_arg {
      Some(v) => match &v.node {
        ConstantNode::Boolean(b) => Ok(Some(*b)),
        _ => Err(InputFilesError::HasProbabilityNotBoolean {
          loc: v.location().clone(),
        }),
      },
      None => Ok(None),
    }
  }

  /// Assumption: Assumes attr is of `file`
  pub fn process_attr(&self, attr: &Attribute) -> Result<InputFile, InputFilesError> {
    if attr.num_pos_args() > 0 {
      let arg = attr.pos_arg(0).unwrap();
      match &arg.node {
        ConstantNode::String(s) => {
          let path = PathBuf::from(s);
          match path.extension() {
            Some(s) if s == "csv" => {
              let deliminator = self.process_deliminator(attr.kw_arg("deliminator"))?;
              let has_header = self.process_has_header(attr.kw_arg("has_header"))?;
              let has_probability = self.process_has_probability(attr.kw_arg("has_probability"))?;
              let input_file = InputFile::csv_with_options(path, deliminator, has_header, has_probability);
              Ok(input_file)
            }
            Some(s) if s == "txt" => Ok(InputFile::Txt(path)),
            Some(s) => Err(InputFilesError::UnknownExtension {
              ext: String::from(s.to_str().unwrap()),
              attr_arg_loc: arg.location().clone(),
            }),
            None => Err(InputFilesError::NoExtension {
              attr_arg_loc: arg.location().clone(),
            }),
          }
        }
        _ => Err(InputFilesError::InvalidArgument {
          attr_arg_loc: arg.location().clone(),
        }),
      }
    } else {
      Err(InputFilesError::InvalidNumAttrArgument {
        actual_num_args: attr.num_pos_args(),
        attr_loc: attr.location().clone(),
      })
    }
  }

  pub fn process_attrs(&mut self, pred: &str, attrs: &Attributes) {
    if let Some(attr) = attrs.iter().find(|a| a.name() == "file") {
      match self.process_attr(attr) {
        Ok(input_file) => {
          self.input_files.insert(pred.to_string(), input_file);
        }
        Err(err) => {
          self.errors.push(err);
        }
      }
    }
  }
}

impl NodeVisitor for InputFilesAnalysis {
  fn visit_input_decl(&mut self, input_decl: &InputDecl) {
    self.process_attrs(input_decl.predicate(), input_decl.attributes());
  }

  fn visit_relation_type_decl(&mut self, relation_type_decl: &RelationTypeDecl) {
    self.process_attrs(relation_type_decl.predicate(), relation_type_decl.attributes());
  }
}

#[derive(Clone, Debug)]
pub enum InputFilesError {
  InvalidNumAttrArgument {
    actual_num_args: usize,
    attr_loc: AstNodeLocation,
  },
  InvalidArgument {
    attr_arg_loc: AstNodeLocation,
  },
  NoExtension {
    attr_arg_loc: AstNodeLocation,
  },
  UnknownExtension {
    ext: String,
    attr_arg_loc: AstNodeLocation,
  },
  HasProbabilityNotBoolean {
    loc: AstNodeLocation,
  },
  HasHeaderNotBoolean {
    loc: AstNodeLocation,
  },
  DeliminatorNotString {
    loc: AstNodeLocation,
  },
  DeliminatorNotSingleCharacter {
    loc: AstNodeLocation,
  },
  DeliminatorNotASCII {
    loc: AstNodeLocation,
  },
}

impl From<InputFilesError> for FrontCompileError {
  fn from(e: InputFilesError) -> Self {
    Self::InputFilesError(e)
  }
}

impl std::fmt::Display for InputFilesError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::InvalidNumAttrArgument {
        actual_num_args,
        attr_loc,
      } => f.write_fmt(format_args!(
        "{}Invalid number attributes of @file attribute. Expected 1, Found {}",
        attr_loc.error_prefix(),
        actual_num_args,
      )),
      Self::InvalidArgument { attr_arg_loc } => f.write_fmt(format_args!(
        "{}Invalid argument of @file attribute. Expected String, found",
        attr_arg_loc.error_prefix()
      )),
      Self::NoExtension { attr_arg_loc } => f.write_fmt(format_args!(
        "{}Input file name does not have an extension",
        attr_arg_loc.error_prefix()
      )),
      Self::UnknownExtension { ext, attr_arg_loc } => f.write_fmt(format_args!(
        "{}Unknown input file extension `.{}`. Expected one from [`.csv`, `.txt`]",
        attr_arg_loc.error_prefix(),
        ext,
      )),
      Self::HasProbabilityNotBoolean { loc } => f.write_fmt(format_args!(
        "{}`has_probability` attribute is not a boolean",
        loc.error_prefix()
      )),
      Self::HasHeaderNotBoolean { loc } => f.write_fmt(format_args!(
        "{}`has_header` attribute is not a boolean",
        loc.error_prefix()
      )),
      Self::DeliminatorNotString { loc } => f.write_fmt(format_args!(
        "{}`deliminator` attribute is not a string",
        loc.error_prefix()
      )),
      Self::DeliminatorNotSingleCharacter { loc } => f.write_fmt(format_args!(
        "{}`deliminator` attribute is not a single character string",
        loc.error_prefix()
      )),
      Self::DeliminatorNotASCII { loc } => f.write_fmt(format_args!(
        "{}`deliminator` attribute is not an ASCII character",
        loc.error_prefix()
      )),
    }
  }
}

impl InputFilesError {
  pub fn report(&self, src: &Sources) {
    match self {
      Self::InvalidNumAttrArgument {
        actual_num_args,
        attr_loc,
      } => {
        println!(
          "Invalid number attributes of @file attribute. Expected 1, Found {}",
          actual_num_args
        );
        attr_loc.report(src);
      }
      Self::InvalidArgument { attr_arg_loc } => {
        println!("Invalid argument of @file attribute. Expected String, found");
        attr_arg_loc.report(src);
      }
      Self::NoExtension { attr_arg_loc } => {
        println!("Input file name does not have an extension");
        attr_arg_loc.report(src);
      }
      Self::UnknownExtension { ext, attr_arg_loc } => {
        println!(
          "Unknown input file extension `.{}`. Expected one from [`.csv`, `.txt`]",
          ext
        );
        attr_arg_loc.report(src);
      }
      Self::HasProbabilityNotBoolean { loc } => {
        println!("`has_probability` attribute is not a boolean");
        loc.report(src);
      }
      Self::HasHeaderNotBoolean { loc } => {
        println!("`has_header` attribute is not a boolean");
        loc.report(src);
      }
      Self::DeliminatorNotString { loc } => {
        println!("`deliminator` attribute is not a string");
        loc.report(src);
      }
      Self::DeliminatorNotSingleCharacter { loc } => {
        println!("`deliminator` attribute is not a single character string");
        loc.report(src);
      }
      Self::DeliminatorNotASCII { loc } => {
        println!("`deliminator` attribute is not an ASCII character");
        loc.report(src);
      }
    }
  }
}
