use std::collections::*;
use std::path::*;

use crate::compiler::ram;
use crate::runtime::error::*;
use crate::utils::*;

use super::*;

#[derive(Debug)]
pub struct InputFileRegistry<Ptr: PointerFamily = ArcFamily> {
  pub input_files: Ptr::Rc<HashMap<PathBuf, InputFileContent>>,
}

impl Clone for InputFileRegistry<ArcFamily> {
  fn clone(&self) -> Self {
    Self {
      input_files: ArcFamily::clone_rc(&self.input_files),
    }
  }
}

impl Clone for InputFileRegistry<RcFamily> {
  fn clone(&self) -> Self {
    Self {
      input_files: RcFamily::clone_rc(&self.input_files),
    }
  }
}

impl<Ptr: PointerFamily> InputFileRegistry<Ptr> {
  pub fn new() -> Self {
    Self {
      input_files: Ptr::new_rc(HashMap::new()),
    }
  }

  pub fn load(&mut self, program: &ram::Program) -> Result<(), IOError> {
    // Iterate through all the input files in the program
    program.input_files().try_for_each(|input_file| {
      if Ptr::get_rc(&self.input_files).contains_key(input_file.file_path()) {
        // Do nothing; the file is already loaded
        Ok(())
      } else {
        // Load the file first; will fail if error happens
        let input_file_content = InputFileContent::load(input_file)?;

        // Insert into the registry
        Ptr::get_rc_mut(&mut self.input_files).insert(input_file.file_path().to_path_buf(), input_file_content);

        // Success
        Ok(())
      }
    })
  }

  pub fn get(&self, file_path: &PathBuf) -> Option<&InputFileContent> {
    Ptr::get_rc(&self.input_files).get(file_path)
  }
}
