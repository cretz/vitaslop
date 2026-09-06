//! `param.sfo` - the key/value blob every Vita app container carries at
//! `sce_sys/param.sfo`, holding the title id and the display title.
//!
//! # Why this exists, when nothing here needs it to boot a title
//! It is the title's own IDENTITY, and identity is what a SAVE is keyed on. Naming a save
//! after the directory the game was extracted into works exactly until two titles are
//! extracted into directories with the same last component - which the common layout
//! (`<name>/extracted/`) guarantees for every title at once. The container answers the
//! question the path only guesses at, and it answers it the same way the browser's launcher
//! does, so a save written by one backend lands where the other looks for it.
//!
//! # The layout
//! Little-endian throughout. A 20-byte header: the magic `\0PSF`, a version word, the byte
//! offset of the KEY table, the byte offset of the DATA table, and the entry count. Then
//! `count` 16-byte index entries, each holding the key's offset within the key table, a
//! format word, the used and reserved data lengths, and the value's offset within the data
//! table. Keys are NUL-terminated ASCII; a value's format word says whether it is text or a
//! 32-bit integer.
//!
//! VERIFIED against the `param.sfo` of five different retail containers, which is the only
//! reason it is written as fact: each parsed to the title id its directory is named for.

/// Format word for a NUL-terminated UTF-8 value. (The other text format is a non-terminated
/// one; both are read the same way here, since trailing NULs are trimmed either way.)
const FMT_TEXT: u16 = 0x0204;
const FMT_TEXT_RAW: u16 = 0x0004;

/// The value of `key`, if the blob is a `param.sfo` and carries that key as text.
///
/// Returns `None` for anything malformed rather than erroring: this is used to LABEL a
/// container, and a caller that cannot read a label falls back to one it can compute. A
/// truncated or foreign file must not be able to panic the run that opened it, so every
/// offset is bounds-checked against the blob rather than trusted.
pub fn text_field(blob: &[u8], key: &str) -> Option<String> {
    if blob.len() < 20 || &blob[..4] != b"\0PSF" {
        return None;
    }
    let word = |off: usize| -> Option<u32> {
        blob.get(off..off + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let key_table = word(8)? as usize;
    let data_table = word(12)? as usize;
    let count = word(16)? as usize;
    for i in 0..count {
        let e = 20 + i * 16;
        let entry = blob.get(e..e + 16)?;
        // CHECKED, not `+`: usize is 32 bits on wasm32, these offsets come out of a game
        // file, and a wrapping add would index somewhere real instead of failing.
        let key_off = key_table.checked_add(u16::from_le_bytes([entry[0], entry[1]]) as usize)?;
        let fmt = u16::from_le_bytes([entry[2], entry[3]]);
        let len = u32::from_le_bytes([entry[4], entry[5], entry[6], entry[7]]) as usize;
        let data_off = data_table
            .checked_add(u32::from_le_bytes([entry[12], entry[13], entry[14], entry[15]]) as usize)?;

        let rest = blob.get(key_off..)?;
        let name_len = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
        if std::str::from_utf8(&rest[..name_len]).ok()? != key {
            continue;
        }
        if fmt != FMT_TEXT && fmt != FMT_TEXT_RAW {
            return None;
        }
        let raw = blob.get(data_off..data_off.checked_add(len)?)?;
        let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        // Lossy: a title's display name carries the vendor's own registered-trademark
        // glyphs, and one byte a decoder dislikes must not cost the whole field.
        return Some(String::from_utf8_lossy(&raw[..end]).into_owned());
    }
    None
}

/// The container's title id (e.g. `PCSA00009`), sanitised to the characters that are safe in
/// a path component - it names a DIRECTORY on both backends, and a value read out of a game
/// file must not be able to name a directory somewhere else.
pub fn title_id(blob: &[u8]) -> Option<String> {
    let raw = text_field(blob, "TITLE_ID")?;
    let clean: String = raw.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect();
    (!clean.is_empty()).then_some(clean)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `param.sfo` with the given text entries, so the reader is tested against the
    /// layout rather than against one captured file (which the repo does not carry).
    fn sfo(entries: &[(&str, &str)]) -> Vec<u8> {
        let mut keys = Vec::new();
        let mut data = Vec::new();
        let mut index = Vec::new();
        for (k, v) in entries {
            let key_off = keys.len() as u16;
            keys.extend_from_slice(k.as_bytes());
            keys.push(0);
            let data_off = data.len() as u32;
            let mut val = v.as_bytes().to_vec();
            val.push(0);
            let len = val.len() as u32;
            data.extend_from_slice(&val);
            index.extend_from_slice(&key_off.to_le_bytes());
            index.extend_from_slice(&FMT_TEXT.to_le_bytes());
            index.extend_from_slice(&len.to_le_bytes());
            index.extend_from_slice(&len.to_le_bytes());
            index.extend_from_slice(&data_off.to_le_bytes());
        }
        let key_table = 20 + index.len();
        let data_table = key_table + keys.len();
        let mut out = Vec::new();
        out.extend_from_slice(b"\0PSF");
        out.extend_from_slice(&0x0101_0000u32.to_le_bytes());
        out.extend_from_slice(&(key_table as u32).to_le_bytes());
        out.extend_from_slice(&(data_table as u32).to_le_bytes());
        out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        out.extend_from_slice(&index);
        out.extend_from_slice(&keys);
        out.extend_from_slice(&data);
        out
    }

    #[test]
    fn a_title_id_is_read_out_of_the_container() {
        // Key order matters: the real files put TITLE_ID last of the four, so a reader that
        // only ever looks at entry 0 would pass a one-entry test and fail every real file.
        let blob = sfo(&[
            ("CATEGORY", "gd"),
            ("STITLE", "A Game"),
            ("TITLE", "A Game, At Length"),
            ("TITLE_ID", "PCSA00000"),
        ]);
        assert_eq!(title_id(&blob).as_deref(), Some("PCSA00000"));
        assert_eq!(text_field(&blob, "TITLE").as_deref(), Some("A Game, At Length"));
        assert_eq!(text_field(&blob, "NOT_A_KEY"), None);
    }

    #[test]
    fn a_title_id_can_never_name_a_directory_of_its_own_choosing() {
        // The value comes out of a game file, and it names a directory on both backends.
        assert_eq!(title_id(&sfo(&[("TITLE_ID", "../../etc")])).as_deref(), Some("etc"));
        assert_eq!(title_id(&sfo(&[("TITLE_ID", "a/b\\c:d")])).as_deref(), Some("abcd"));
        assert_eq!(title_id(&sfo(&[("TITLE_ID", "///")])), None, "nothing usable is left");
    }

    #[test]
    fn nothing_malformed_can_panic_the_reader() {
        assert_eq!(title_id(b""), None);
        assert_eq!(title_id(b"\0PSFnot-really-a-container"), None);
        let good = sfo(&[("TITLE_ID", "PCSA00000")]);
        // Every truncation of a real one. A container is a game file; a run must not die
        // because one was cut short.
        for n in 0..good.len() {
            let _ = title_id(&good[..n]);
        }
        // ...and a header that points its tables off the end of the blob.
        let mut bad = good.clone();
        bad[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(title_id(&bad), None);
        let mut bad = good.clone();
        bad[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(title_id(&bad), None);
        // A COUNT of four billion entries. This one still ANSWERS - entry 0 is intact and
        // carries the id - and that is correct: the reader stops at the first entry that
        // runs off the end rather than trusting the count, so a bogus count costs nothing.
        // The property being pinned is that it returns at all.
        let mut bad = good;
        bad[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(title_id(&bad).as_deref(), Some("PCSA00000"));
    }
}
