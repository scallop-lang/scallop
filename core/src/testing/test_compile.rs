use crate::compiler;

pub fn expect_compile(s: &str) {
  compiler::compile_string_to_ram(s.to_string()).expect("Compile Failure");
}

pub fn expect_compile_failure<F>(s: &str, f: F)
where
  F: Fn(compiler::CompileError) -> bool,
{
  match compiler::compile_string_to_ram(s.to_string()) {
    Ok(_) => panic!("Compilation passed; expected failure"),
    Err(es) => {
      for e in es {
        if f(e) {
          return;
        }
      }
      panic!("Expected failure not found")
    }
  }
}

pub fn expect_front_compile_failure<F>(s: &str, f: F)
where
  F: Fn(String) -> bool,
{
  match compiler::compile_string_to_ram(s.to_string()) {
    Ok(_) => panic!("Compilation passed; expected failure"),
    Err(es) => {
      for e in es {
        match e {
          compiler::CompileError::Front(e) => {
            let e = format!("{}", e);
            if f(e) {
              return;
            }
          }
          _ => {}
        }
      }
      panic!("Expected failure not found")
    }
  }
}
