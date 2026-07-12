//! Binary-side command implementations, split out of main.rs. These are
//! child modules of the binary crate root, so they see main.rs's private
//! args structs and harness helpers via `use crate::*` — command moves stay
//! verbatim, with `pub(crate)` on the entry points only.

pub(crate) mod diag;
pub(crate) mod dspark;
pub(crate) mod quant;
pub(crate) mod verify;
