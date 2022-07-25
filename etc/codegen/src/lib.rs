#![feature(proc_macro_span)]

use proc_macro::TokenStream;
use quote::quote;

use scallop_core::*;

#[proc_macro]
pub fn scallop(tokens: TokenStream) -> TokenStream {
  let opt = compiler::CompileOptions::default();

  let src = compiler::front::RustMacroSource::new(tokens.into());
  let ram = match compiler::compile_source_to_ram(src) {
    Ok(ram) => ram,
    Err(errs) => {
      for err in errs {
        println!("{}", err);
      }
      return quote! {}.into();
    }
  };

  // Generate related rust code
  let rs_mod = ram.to_rs_module(&opt);
  let rs_create_edb_fn = ram.to_rs_create_edb_fn();

  quote! {
    #rs_mod
    #rs_create_edb_fn
  }
  .into()
}
