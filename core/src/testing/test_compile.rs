use crate::compiler;

/// Expect the given program compiles; panics if there is compilation failure
pub fn expect_compile(s: &str) {
  compiler::compile_string_to_ram(s.to_string()).expect("Compile Failure");
}

/// Expect the given program fails to compile; panics if the compilation succeed
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

/// Expect the given program fails to compile in the FRONT compilation stage
///
/// The given `f` takes in an error `String` and returns whether that string
/// represents a particular error that the user expected.
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

/// Expect the given program fails to compile in the FRONT compilation stage
///
/// The given `f` takes in an error `String` and returns whether that string
/// represents a particular error that the user expected.
pub fn expect_front_compile_failure_with_modifier<M, F>(s: &str, m: M, f: F)
where
  M: Fn(&mut compiler::front::FrontContext),
  F: Fn(String) -> bool,
{
  let mut ctx = compiler::front::FrontContext::new();
  m(&mut ctx);
  match ctx.compile_string(s.to_string()) {
    Ok(_) => {
      panic!("Compilation passed; expected failure")
    }
    Err(err) => {
      let err = format!("{}", err);
      if !f(err) {
        panic!("Expected failure not found")
      }
    }
  }
}
