//! The record the library keeps per imported title, written at import from the
//! ingest probe and read every time the library is drawn.
//!
//! Small on purpose: the images live beside it as files (`icon0.png`, `pic0.png`),
//! not inside it, so listing a thousand titles reads a thousand short JSON files
//! and no image bytes.

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default, rename_all = "camelCase")]
pub struct TitleMeta {
    /// `PCSA00000`-shaped - the key everything else is stored under.
    pub title_id: String,
    /// `param.sfo` `TITLE`, or the id when there is none.
    pub title: String,
    pub content_id: String,
    pub app_version: String,
    /// What it was imported from: `pkg`, `pfs`, `dump`.
    pub source_kind: String,
    /// Bytes on disk after decryption.
    pub bytes: u64,
    pub files: u32,
    pub has_icon: bool,
    pub has_pic: bool,
    /// Unix milliseconds.
    pub imported_at: u64,
    /// Unix milliseconds of the last run, 0 if never.
    pub last_played_at: u64,
}

impl TitleMeta {
    /// The lower-cased haystack a library search matches against.
    pub fn search_key(&self) -> String {
        format!("{} {} {}", self.title, self.title_id, self.content_id).to_lowercase()
    }
}

/// The record kept per installed add-on, beside its decrypted files. An add-on belongs to
/// one title and lives and dies with it: listed, added and removed on that title's page.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default, rename_all = "camelCase")]
pub struct AddonMeta {
    /// The title it belongs to.
    pub title_id: String,
    /// Its directory name, the label that ends its content id - what the title asks for.
    pub dir: String,
    pub content_id: String,
    /// `param.sfo` `TITLE`, or the directory name when there is none.
    pub title: String,
    /// `param.sfo` `VERSION`.
    pub version: String,
    /// What it was imported from: `pkg`, `pfs`, `dump`.
    pub source_kind: String,
    pub bytes: u64,
    pub files: u32,
    /// Unix milliseconds.
    pub imported_at: u64,
    /// Turned off on the title page: kept, but the title does not see it.
    pub disabled: bool,
}

/// The record kept beside a title's rePatch overlay (see `vitaslop_runtime::ingest::repatch`),
/// written last. One per title; turned off, replaced or removed without touching the title.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default, rename_all = "camelCase")]
pub struct RepatchMeta {
    /// The title it lays over.
    pub title_id: String,
    /// What it was added from: the picked folder's or zip's name.
    pub source: String,
    pub bytes: u64,
    pub files: u32,
    /// The executables it replaces or adds (`eboot.bin`, `sce_module/...`).
    pub code: Vec<String>,
    /// The add-on directories it lays files over.
    pub addcont: Vec<String>,
    /// The compile caches' name for its code, empty when it carries none.
    pub code_hash: String,
    /// Unix milliseconds.
    pub imported_at: u64,
    /// Turned off on the title page: kept, but the title runs without it.
    pub disabled: bool,
}
