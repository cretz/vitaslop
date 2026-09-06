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
    /// `PCSE00120` - the key everything else is stored under.
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
