//! Add-ons ("additional content", `addcont`): the facts every front end needs to file one.
//!
//! An add-on arrives in exactly the containers a title does (a pkg + `work.bin`, a zip, a
//! decrypted folder) and decrypts through the same pipeline; what makes it an add-on is its
//! `param.sfo` `CATEGORY` of `ac`. It has no executable. A device installs it at
//! `ux0:addcont/<title id>/<dir>/` and the title reads it through `addcont0:<dir>/...`, where
//! `<dir>` is the label that ends its content id.

/// The `param.sfo` `CATEGORY` an add-on carries.
pub const CATEGORY: &str = "ac";

/// Whether a `param.sfo` `CATEGORY` names an add-on.
pub fn is_addcont_category(category: &str) -> bool {
    category.trim().eq_ignore_ascii_case(CATEGORY)
}

/// The add-on's directory name: the 16-character label after the last `-` of a content id
/// `XX0000-ABCD12345_00-LABEL0123456789`. `None` for a content id of any other shape, which
/// is refused rather than guessed at - the name is what the title asks `sceAppUtilDrmOpen`
/// for, and a wrong one is an add-on that silently never appears.
pub fn dir_name(content_id: &str) -> Option<&str> {
    let label = content_id.rsplit_once('-')?.1;
    (label.len() == 16
        && label.bytes().all(|b| b.is_ascii_alphanumeric())
        && title_id(content_id).is_some())
    .then_some(label)
}

/// The title id a content id belongs to (`ABCD12345` in `XX0000-ABCD12345_00-...`).
pub fn title_id(content_id: &str) -> Option<&str> {
    let id = content_id.get(7..16)?;
    (content_id.as_bytes().get(6) == Some(&b'-') && id.bytes().all(|b| b.is_ascii_alphanumeric()))
        .then_some(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_content_id_names_its_title_and_directory() {
        let cid = "XX0000-ABCD12345_00-LABEL01234567890";
        assert_eq!(title_id(cid), Some("ABCD12345"));
        assert_eq!(dir_name(cid), Some("LABEL01234567890"));
        assert_eq!(
            dir_name("XX0000-ABCD12345_00-LABEL012345678901"),
            None,
            "a 17-character label"
        );
        assert_eq!(dir_name("XX0000-ABCD12345_00-SHORT"), None);
        assert_eq!(dir_name("garbage"), None);
        assert_eq!(title_id("garbage"), None);
    }

    #[test]
    fn the_category_is_matched_loosely() {
        assert!(is_addcont_category("ac"));
        assert!(is_addcont_category("AC "));
        assert!(!is_addcont_category("gd"));
    }
}
