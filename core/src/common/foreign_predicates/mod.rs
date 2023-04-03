//! Library of foreign predicates

use std::convert::*;

use crate::utils::*;

use super::input_tag::*;
use super::foreign_predicate::*;
use super::value::*;
use super::value_type::*;

mod float_eq;
mod range;
mod string_chars;
mod soft_cmp;
mod soft_eq;
mod soft_gt;
mod soft_lt;
mod soft_neq;

pub use float_eq::*;
pub use range::*;
pub use string_chars::*;
pub use soft_cmp::*;
pub use soft_eq::*;
pub use soft_gt::*;
pub use soft_lt::*;
pub use soft_neq::*;
