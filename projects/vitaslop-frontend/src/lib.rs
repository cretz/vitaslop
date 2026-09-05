//! What the browser and the desktop front ends share.
//!
//! - [`settings`]: the one settings record, its defaults, and the merge that turns a
//!   global record plus a per-title patch into the settings a run uses.
//! - [`input`]: the Vita's buttons by name and bit, and the default keyboard and
//!   gamepad maps in the W3C vocabularies both platforms can speak.
//! - [`meta`]: the record the library keeps per imported title.
//!
//! Everything is `serde` JSON, because JSON is what crosses to the page and what
//! the desktop writes beside its library.

pub mod input;
pub mod meta;
pub mod settings;
