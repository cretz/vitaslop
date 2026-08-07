//! A title's own trophy set, read from the `TROPHY.TRP` it ships.
//!
//! Every trophy fact a game can ask `SceNpTrophy` for - how many trophies exist, their
//! grades, their localized names and descriptions, and their icons - is shipped inside the
//! title at `sce_sys/trophy/<NPCOMMID>/TROPHY.TRP`. Nothing here is invented: the counts a
//! guest reads come from the guest's own data. What a console adds on top is the per-account
//! UNLOCK ledger, which for a fresh offline profile is empty and grows only as the title
//! unlocks trophies during the run (see `VitaState`'s trophy state).
//!
//! The TRP container is a big-endian header plus a fixed-size entry table:
//!
//! ```text
//! 0x00 u32  magic 0xDCA24D00
//! 0x04 u32  version (2)
//! 0x08 u64  total file size
//! 0x10 u32  entry count
//! 0x14 u32  entry size (0x40)
//! 0x18 u32  dev flag
//! 0x1C      SHA-1 over the file, then padding to `header size`
//!
//! entry:
//! 0x00 char[32] name, NUL padded
//! 0x20 u64      offset from the start of the file
//! 0x28 u64      size
//! 0x30 u32      flags (1 = the entry carries an Sce-Np-Trophy-Signature)
//! 0x34          padding to 0x40
//! ```
//!
//! The entries are `TROPCONF.SFM` (the grade/id table), `TROP.SFM` plus `TROP_<lang>.SFM`
//! (the same table with localized names and descriptions), `ICON0.PNG` (the set icon),
//! `TROP<nnn>.PNG` (per trophy) and `GR<nnn>.PNG` (per group). The `.SFM` files are plain
//! XML - the leading `<!--Sce-Np-Trophy-Signature: ...-->` comment is a signature over the
//! content, not encryption, so no key chain is involved in reading any of this.
//!
//! Note the whole file is PFS-encrypted inside a retail app, so it only looks like noise
//! until `ingest` has decrypted it; by the time the guest filesystem holds it, it is plain.

use std::fmt;

/// `SCE_NP_TROPHY_GRADE_*`. The numbering is the API's, not the file's - `TROPCONF.SFM`
/// spells grades as the single letters `P`/`G`/`S`/`B`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grade {
    Platinum = 1,
    Gold = 2,
    Silver = 3,
    Bronze = 4,
}

impl Grade {
    /// The `ttype` attribute letter used in `TROPCONF.SFM`.
    fn from_ttype(s: &str) -> Option<Grade> {
        match s {
            "P" => Some(Grade::Platinum),
            "G" => Some(Grade::Gold),
            "S" => Some(Grade::Silver),
            "B" => Some(Grade::Bronze),
            _ => None,
        }
    }
}

/// Why a trophy set could not be read. Each variant maps to the console's own
/// `SCE_NP_TROPHY_ERROR_*` code at the API boundary, so a title sees the error a real
/// system would raise for the same broken file rather than a generic failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrophyError {
    /// The file does not start with the TRP magic.
    NotTrp,
    /// A TRP whose container version this reader does not implement.
    UnsupportedVersion(u32),
    /// The header or an entry's byte range runs past the end of the file.
    Truncated(&'static str),
    /// No `TROPCONF.SFM` entry, so the set has no grade table.
    MissingConf,
    /// The `.SFM` XML did not parse, or did not hold what a trophy conf must hold.
    BadConf(String),
}

impl fmt::Display for TrophyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrophyError::NotTrp => write!(f, "not a TRP container (bad magic)"),
            TrophyError::UnsupportedVersion(v) => write!(f, "unsupported TRP version {v}"),
            TrophyError::Truncated(what) => write!(f, "TRP truncated: {what}"),
            TrophyError::MissingConf => write!(f, "TRP has no TROPCONF.SFM"),
            TrophyError::BadConf(why) => write!(f, "bad trophy conf: {why}"),
        }
    }
}

/// One file inside a TRP container.
#[derive(Clone, Debug)]
struct TrpEntry {
    name: String,
    offset: usize,
    size: usize,
}

/// A parsed TRP container: the whole file plus its entry table. Entry bytes are borrowed
/// from `data`, so extracting an icon costs a slice rather than a copy.
#[derive(Debug)]
pub struct TrpArchive {
    data: Vec<u8>,
    entries: Vec<TrpEntry>,
}

const TRP_MAGIC: u32 = 0xDCA2_4D00;
/// The only container version seen in the wild, and the one this layout describes.
const TRP_VERSION: u32 = 2;
/// The header fields this reader needs end at 0x1C; the entry table starts at the offset the
/// header's own `entry size` implies, which for every observed file is 0x40.
const TRP_HEADER_SIZE: usize = 0x40;
const TRP_ENTRY_SIZE: usize = 0x40;

impl TrpArchive {
    /// Parse a TRP container. Every offset is bounds-checked against the buffer, so a
    /// truncated or malformed file is an error rather than a panic.
    pub fn parse(data: Vec<u8>) -> Result<TrpArchive, TrophyError> {
        if data.len() < TRP_HEADER_SIZE {
            return Err(TrophyError::Truncated("header"));
        }
        if be_u32(&data, 0) != TRP_MAGIC {
            return Err(TrophyError::NotTrp);
        }
        let version = be_u32(&data, 4);
        if version != TRP_VERSION {
            return Err(TrophyError::UnsupportedVersion(version));
        }
        let count = be_u32(&data, 0x10) as usize;
        let entry_size = be_u32(&data, 0x14) as usize;
        if entry_size != TRP_ENTRY_SIZE {
            return Err(TrophyError::UnsupportedVersion(version));
        }
        let table_end = TRP_HEADER_SIZE
            .checked_add(count.checked_mul(entry_size).ok_or(TrophyError::Truncated("entry table"))?)
            .ok_or(TrophyError::Truncated("entry table"))?;
        if table_end > data.len() {
            return Err(TrophyError::Truncated("entry table"));
        }

        let mut entries = Vec::with_capacity(count);
        for i in 0..count {
            let base = TRP_HEADER_SIZE + i * entry_size;
            let name_bytes = &data[base..base + 32];
            let name_end = name_bytes.iter().position(|&b| b == 0).unwrap_or(32);
            let name = String::from_utf8_lossy(&name_bytes[..name_end]).into_owned();
            let offset = be_u64(&data, base + 32) as usize;
            let size = be_u64(&data, base + 40) as usize;
            let end = offset.checked_add(size).ok_or(TrophyError::Truncated("entry range"))?;
            if end > data.len() {
                return Err(TrophyError::Truncated("entry range"));
            }
            entries.push(TrpEntry { name, offset, size });
        }
        Ok(TrpArchive { data, entries })
    }

    /// The bytes of the named entry (case-insensitive, as the container's own names are
    /// upper case and callers spell them either way), or `None` if it is absent.
    pub fn file(&self, name: &str) -> Option<&[u8]> {
        self.entries
            .iter()
            .find(|e| e.name.eq_ignore_ascii_case(name))
            .map(|e| &self.data[e.offset..e.offset + e.size])
    }

    /// Every entry name, in table order. Used by diagnostics and by the localized-conf
    /// search.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|e| e.name.as_str())
    }
}

/// One trophy, as the title's own conf declares it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrophyDef {
    pub id: u32,
    pub grade: Grade,
    /// The `pid` attribute: the group this trophy belongs to, or -1 for none.
    pub group_id: i32,
    pub hidden: bool,
    pub name: String,
    pub detail: String,
}

/// One trophy group (a conf with no `<group>` elements has none).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupDef {
    pub id: i32,
    pub name: String,
    pub detail: String,
}

/// A title's complete trophy set: what the conf declares, plus the container the icons
/// come from.
#[derive(Debug)]
pub struct TrophySet {
    /// `<npcommid>`, e.g. the `NPWR#####_00` that also names the containing directory.
    pub comm_id: String,
    /// `<title-name>` - absent from `TROPCONF.SFM`, present in the localized confs.
    pub title: String,
    /// `<title-detail>`.
    pub detail: String,
    pub groups: Vec<GroupDef>,
    /// Trophies in conf order (which is id order in every observed file).
    pub trophies: Vec<TrophyDef>,
    archive: TrpArchive,
}

impl TrophySet {
    /// Read a set from TRP bytes, taking names and descriptions from the conf localized for
    /// `lang` (an `SCE_SYSTEM_PARAM_LANG_*` value) when the container ships one.
    ///
    /// `TROPCONF.SFM` is the authority for which trophies exist and what grade each is; it
    /// carries no display text. `TROP_<lang>.SFM`, or `TROP.SFM` as the default, repeats the
    /// same table with the text filled in. Taking ids and grades from `TROPCONF.SFM` and
    /// merging text in by id means a localized conf that disagrees about the set cannot
    /// change what the set IS.
    pub fn parse(data: Vec<u8>, lang: u32) -> Result<TrophySet, TrophyError> {
        let archive = TrpArchive::parse(data)?;
        let conf_bytes = archive.file("TROPCONF.SFM").ok_or(TrophyError::MissingConf)?;
        let conf = Conf::parse(conf_bytes)?;

        let localized_name = localized_conf_name(&archive, lang);
        let localized = match localized_name.as_deref().and_then(|n| archive.file(n)) {
            Some(bytes) => Some(Conf::parse(bytes)?),
            None => None,
        };

        let (title, detail) = match &localized {
            Some(l) => (l.title.clone(), l.detail.clone()),
            None => (conf.title.clone(), conf.detail.clone()),
        };

        let text_for = |id: u32| -> (String, String) {
            localized
                .as_ref()
                .and_then(|l| l.trophies.iter().find(|t| t.id == id))
                .map(|t| (t.name.clone(), t.detail.clone()))
                .unwrap_or_default()
        };
        let trophies = conf
            .trophies
            .iter()
            .map(|t| {
                let (name, detail) = text_for(t.id);
                TrophyDef { id: t.id, grade: t.grade, group_id: t.group_id, hidden: t.hidden, name, detail }
            })
            .collect();

        let group_text = |id: i32| -> (String, String) {
            localized
                .as_ref()
                .and_then(|l| l.groups.iter().find(|g| g.id == id))
                .map(|g| (g.name.clone(), g.detail.clone()))
                .unwrap_or_default()
        };
        let groups = conf
            .groups
            .iter()
            .map(|g| {
                let (name, detail) = group_text(g.id);
                GroupDef { id: g.id, name, detail }
            })
            .collect();

        Ok(TrophySet { comm_id: conf.comm_id, title, detail, groups, trophies, archive })
    }

    /// The trophy with this id, or `None` if the set does not declare it.
    pub fn trophy(&self, id: u32) -> Option<&TrophyDef> {
        self.trophies.iter().find(|t| t.id == id)
    }

    /// The set's own icon (`ICON0.PNG`).
    pub fn game_icon(&self) -> Option<&[u8]> {
        self.archive.file("ICON0.PNG")
    }

    /// One trophy's icon (`TROP<nnn>.PNG`).
    pub fn trophy_icon(&self, id: u32) -> Option<&[u8]> {
        self.archive.file(&format!("TROP{id:03}.PNG"))
    }

    /// One group's icon (`GR<nnn>.PNG`).
    pub fn group_icon(&self, id: i32) -> Option<&[u8]> {
        if id < 0 {
            return None;
        }
        self.archive.file(&format!("GR{id:03}.PNG"))
    }

    /// Counts of (total, platinum, gold, silver, bronze) over the trophies matching
    /// `keep` - the same fold serves the whole set and one group.
    pub fn counts(&self, keep: impl Fn(&TrophyDef) -> bool) -> GradeCounts {
        let mut c = GradeCounts::default();
        for t in self.trophies.iter().filter(|t| keep(t)) {
            c.total += 1;
            match t.grade {
                Grade::Platinum => c.platinum += 1,
                Grade::Gold => c.gold += 1,
                Grade::Silver => c.silver += 1,
                Grade::Bronze => c.bronze += 1,
            }
        }
        c
    }
}

/// Trophy counts broken down by grade, as every `SceNpTrophy*Details`/`*Data` struct
/// reports them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GradeCounts {
    pub total: u32,
    pub platinum: u32,
    pub gold: u32,
    pub silver: u32,
    pub bronze: u32,
}

/// The trophy state a run accumulates: the sets a title has opened contexts on, and the
/// per-profile UNLOCK ledger.
///
/// The sets are the title's own shipped data; the ledger is the only part a console adds,
/// and off-console it starts empty (a fresh profile has unlocked nothing) and grows as the
/// title unlocks trophies during the run. Keeping it here rather than fabricating unlock
/// counts means a title's own "you just earned this" path reads back exactly what it
/// unlocked, which is what read-after-write on hardware does.
#[derive(Debug, Default)]
pub struct TrophyStore {
    /// NP communication id -> the set read from that title's TRP. One load per set,
    /// however many contexts are opened on it.
    sets: std::collections::HashMap<String, TrophySet>,
    /// Live context id -> the NP communication id it was created for.
    contexts: std::collections::HashMap<u32, String>,
    /// NP communication id -> the trophy ids unlocked this run and the `SceRtcTick` each
    /// was earned at. The tick is kept because `sceNpTrophyGetTrophyInfo` reports it, and
    /// a title that draws an earned-on date would otherwise get a zero.
    unlocked: std::collections::HashMap<String, std::collections::BTreeMap<u32, u64>>,
}

impl TrophyStore {
    /// Whether a set for this communication id has already been read.
    pub fn has_set(&self, comm_id: &str) -> bool {
        self.sets.contains_key(comm_id)
    }

    /// Record a freshly read set, keyed by its own `<npcommid>`.
    pub fn insert_set(&mut self, set: TrophySet) {
        self.sets.insert(set.comm_id.clone(), set);
    }

    /// Bind a new context id to an already-inserted set.
    pub fn open_context(&mut self, context: u32, comm_id: &str) {
        self.contexts.insert(context, comm_id.to_string());
    }

    /// Drop a context. The set and the unlock ledger outlive it, as they do on hardware -
    /// a title that destroys and recreates a context sees the same unlocks.
    /// Returns whether the context existed.
    pub fn close_context(&mut self, context: u32) -> bool {
        self.contexts.remove(&context).is_some()
    }

    /// The communication id a context was opened for.
    pub fn comm_id_of(&self, context: u32) -> Option<&str> {
        self.contexts.get(&context).map(String::as_str)
    }

    /// The set behind a context, or `None` if the context was never created.
    pub fn set_for(&self, context: u32) -> Option<&TrophySet> {
        self.sets.get(self.contexts.get(&context)?)
    }

    /// Whether a trophy is unlocked in this set's ledger.
    pub fn is_unlocked(&self, comm_id: &str, id: u32) -> bool {
        self.unlocked_at(comm_id, id).is_some()
    }

    /// The `SceRtcTick` a trophy was earned at, or `None` if it is still locked.
    pub fn unlocked_at(&self, comm_id: &str, id: u32) -> Option<u64> {
        self.unlocked.get(comm_id).and_then(|s| s.get(&id)).copied()
    }

    /// Record an unlock at `tick`. Returns false (and keeps the original tick) if it was
    /// already unlocked.
    pub fn unlock(&mut self, comm_id: &str, id: u32, tick: u64) -> bool {
        match self.unlocked.entry(comm_id.to_string()).or_default().entry(id) {
            std::collections::btree_map::Entry::Occupied(_) => false,
            std::collections::btree_map::Entry::Vacant(slot) => {
                slot.insert(tick);
                true
            }
        }
    }

    /// Counts of the unlocked trophies matching `keep`, so unlocked and total counts are
    /// folded by the same rule.
    pub fn unlocked_counts(&self, set: &TrophySet, keep: impl Fn(&TrophyDef) -> bool) -> GradeCounts {
        let ledger = self.unlocked.get(&set.comm_id);
        set.counts(|t| keep(t) && ledger.is_some_and(|l| l.contains_key(&t.id)))
    }
}

/// The localized conf entry to prefer for `lang`: `TROP_<nn>.SFM` when the container ships
/// one for that language, otherwise `TROP.SFM` (the set's default language) when present.
/// The two-digit suffix is the same `SCE_SYSTEM_PARAM_LANG_*` numbering the system uses.
fn localized_conf_name(archive: &TrpArchive, lang: u32) -> Option<String> {
    let preferred = format!("TROP_{lang:02}.SFM");
    if archive.names().any(|n| n.eq_ignore_ascii_case(&preferred)) {
        return Some(preferred);
    }
    archive.names().any(|n| n.eq_ignore_ascii_case("TROP.SFM")).then(|| "TROP.SFM".to_string())
}

/// One parsed `.SFM` conf. `TROPCONF.SFM` and the localized `TROP*.SFM` share this grammar;
/// only the display text differs.
#[derive(Debug, Default)]
struct Conf {
    comm_id: String,
    title: String,
    detail: String,
    groups: Vec<GroupDef>,
    trophies: Vec<TrophyDef>,
}

impl Conf {
    fn parse(bytes: &[u8]) -> Result<Conf, TrophyError> {
        let text = std::str::from_utf8(bytes)
            .map_err(|e| TrophyError::BadConf(format!("conf is not UTF-8: {e}")))?;
        let mut conf = Conf::default();
        // The element currently being filled, so a `<name>`/`<detail>` child is attributed
        // to the right owner. The grammar nests exactly one level below `<trophyconf>`.
        let mut open_trophy: Option<TrophyDef> = None;
        let mut open_group: Option<GroupDef> = None;
        let mut pending_text: Option<&'static str> = None;
        let mut saw_root = false;

        for ev in XmlEvents::new(text) {
            match ev? {
                XmlEvent::Start { name, attrs, self_closing } => {
                    match name {
                        "trophyconf" => saw_root = true,
                        "npcommid" | "title-name" | "title-detail" | "name" | "detail" => {
                            pending_text = Some(match name {
                                "npcommid" => "npcommid",
                                "title-name" => "title-name",
                                "title-detail" => "title-detail",
                                "name" => "name",
                                _ => "detail",
                            });
                        }
                        "group" => {
                            let id = attr_i32(&attrs, "id", name)?;
                            open_group = Some(GroupDef { id, name: String::new(), detail: String::new() });
                        }
                        "trophy" => {
                            let id = attr_u32(&attrs, "id", name)?;
                            let ttype = attr(&attrs, "ttype").ok_or_else(|| {
                                TrophyError::BadConf(format!("<trophy id={id}> has no ttype"))
                            })?;
                            let grade = Grade::from_ttype(ttype).ok_or_else(|| {
                                TrophyError::BadConf(format!("<trophy id={id}> has unknown ttype {ttype:?}"))
                            })?;
                            let hidden = matches!(attr(&attrs, "hidden"), Some("yes"));
                            let group_id = match attr(&attrs, "pid") {
                                Some(_) => attr_i32(&attrs, "pid", name)?,
                                None => -1,
                            };
                            let def = TrophyDef {
                                id,
                                grade,
                                group_id,
                                hidden,
                                name: String::new(),
                                detail: String::new(),
                            };
                            if self_closing {
                                conf.trophies.push(def);
                            } else {
                                open_trophy = Some(def);
                            }
                        }
                        // A conf carries policy elements this reader has no use for
                        // (`trophyset-version`, `parental-level`, ...). Ignoring an
                        // unknown element is safe: the fields that matter are all
                        // required below, so a conf that omits one still fails.
                        _ => {}
                    }
                    if self_closing {
                        pending_text = None;
                        if name == "group" {
                            if let Some(g) = open_group.take() {
                                conf.groups.push(g);
                            }
                        }
                    }
                }
                XmlEvent::Text(text) => {
                    let Some(field) = pending_text.take() else { continue };
                    match (field, open_trophy.as_mut(), open_group.as_mut()) {
                        ("npcommid", _, _) => conf.comm_id = text,
                        ("title-name", _, _) => conf.title = text,
                        ("title-detail", _, _) => conf.detail = text,
                        ("name", Some(t), _) => t.name = text,
                        ("detail", Some(t), _) => t.detail = text,
                        ("name", None, Some(g)) => g.name = text,
                        ("detail", None, Some(g)) => g.detail = text,
                        _ => {}
                    }
                }
                XmlEvent::End { name } => {
                    pending_text = None;
                    match name {
                        "trophy" => {
                            if let Some(t) = open_trophy.take() {
                                conf.trophies.push(t);
                            }
                        }
                        "group" => {
                            if let Some(g) = open_group.take() {
                                conf.groups.push(g);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        if !saw_root {
            return Err(TrophyError::BadConf("no <trophyconf> element".into()));
        }
        if conf.comm_id.is_empty() {
            return Err(TrophyError::BadConf("no <npcommid>".into()));
        }
        Ok(conf)
    }
}

fn attr<'a>(attrs: &'a [(&'a str, String)], key: &str) -> Option<&'a str> {
    attrs.iter().find(|(k, _)| *k == key).map(|(_, v)| v.as_str())
}

fn attr_u32(attrs: &[(&str, String)], key: &str, elem: &str) -> Result<u32, TrophyError> {
    let raw = attr(attrs, key)
        .ok_or_else(|| TrophyError::BadConf(format!("<{elem}> has no {key}")))?;
    raw.parse::<u32>()
        .map_err(|_| TrophyError::BadConf(format!("<{elem}> {key}={raw:?} is not a number")))
}

fn attr_i32(attrs: &[(&str, String)], key: &str, elem: &str) -> Result<i32, TrophyError> {
    let raw = attr(attrs, key)
        .ok_or_else(|| TrophyError::BadConf(format!("<{elem}> has no {key}")))?;
    raw.parse::<i32>()
        .map_err(|_| TrophyError::BadConf(format!("<{elem}> {key}={raw:?} is not a number")))
}

fn be_u32(d: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]])
}

fn be_u64(d: &[u8], at: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&d[at..at + 8]);
    u64::from_be_bytes(b)
}

// ---------------------------------------------------------------------------
// A minimal XML pull reader
//
// The confs are small, machine-generated, and use one narrow slice of XML: elements with
// quoted attributes, character data, comments, and an optional declaration. A dependency
// would be a poor trade for that, but a hand parser has to be strict or it turns a
// malformed file into silently wrong trophy counts - so anything it does not understand is
// an error, never a skip.
// ---------------------------------------------------------------------------

enum XmlEvent<'a> {
    Start { name: &'a str, attrs: Vec<(&'a str, String)>, self_closing: bool },
    End { name: &'a str },
    /// Character data with entity references resolved. Whitespace-only runs (the
    /// indentation between elements) are not emitted.
    Text(String),
}

struct XmlEvents<'a> {
    src: &'a str,
    pos: usize,
    done: bool,
}

impl<'a> XmlEvents<'a> {
    fn new(src: &'a str) -> Self {
        XmlEvents { src, pos: 0, done: false }
    }
}

impl<'a> Iterator for XmlEvents<'a> {
    type Item = Result<XmlEvent<'a>, TrophyError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        loop {
            let rest = &self.src[self.pos..];
            if rest.is_empty() {
                self.done = true;
                return None;
            }
            if let Some(text_len) = rest.find('<') {
                if text_len > 0 {
                    let raw = &rest[..text_len];
                    self.pos += text_len;
                    if !raw.trim().is_empty() {
                        return Some(decode_entities(raw.trim()).map(XmlEvent::Text));
                    }
                    continue;
                }
            } else {
                // Trailing character data after the last element: the confs end with a
                // newline, so only whitespace is acceptable here.
                self.done = true;
                if rest.trim().is_empty() {
                    return None;
                }
                return Some(Err(TrophyError::BadConf("text after the root element".into())));
            }

            // At a '<'. Comments and declarations are skipped wholesale.
            if rest.starts_with("<!--") {
                let Some(end) = rest.find("-->") else {
                    self.done = true;
                    return Some(Err(TrophyError::BadConf("unterminated comment".into())));
                };
                self.pos += end + 3;
                continue;
            }
            if rest.starts_with("<?") || rest.starts_with("<!") {
                let Some(end) = rest.find('>') else {
                    self.done = true;
                    return Some(Err(TrophyError::BadConf("unterminated declaration".into())));
                };
                self.pos += end + 1;
                continue;
            }

            let Some(end) = rest.find('>') else {
                self.done = true;
                return Some(Err(TrophyError::BadConf("unterminated tag".into())));
            };
            let inner = &rest[1..end];
            self.pos += end + 1;

            if let Some(name) = inner.strip_prefix('/') {
                return Some(Ok(XmlEvent::End { name: name.trim() }));
            }
            let (inner, self_closing) = match inner.strip_suffix('/') {
                Some(stripped) => (stripped, true),
                None => (inner, false),
            };
            let mut cut = inner.len();
            for (i, c) in inner.char_indices() {
                if c.is_ascii_whitespace() {
                    cut = i;
                    break;
                }
            }
            let name = &inner[..cut];
            if name.is_empty() {
                self.done = true;
                return Some(Err(TrophyError::BadConf("empty element name".into())));
            }
            return Some(parse_attrs(&inner[cut..]).map(|attrs| XmlEvent::Start {
                name,
                attrs,
                self_closing,
            }));
        }
    }
}

/// Parse `key="value"` pairs. Both quote styles are accepted; an unquoted or unterminated
/// value is an error rather than a guess.
fn parse_attrs(mut s: &str) -> Result<Vec<(&str, String)>, TrophyError> {
    let mut out = Vec::new();
    loop {
        s = s.trim_start();
        if s.is_empty() {
            return Ok(out);
        }
        let Some(eq) = s.find('=') else {
            return Err(TrophyError::BadConf(format!("attribute without a value near {s:?}")));
        };
        let key = s[..eq].trim();
        let after = s[eq + 1..].trim_start();
        let quote = match after.chars().next() {
            Some(q @ ('"' | '\'')) => q,
            _ => return Err(TrophyError::BadConf(format!("unquoted attribute value near {after:?}"))),
        };
        let body = &after[quote.len_utf8()..];
        let Some(close) = body.find(quote) else {
            return Err(TrophyError::BadConf(format!("unterminated attribute value near {after:?}")));
        };
        out.push((key, decode_entities(&body[..close])?));
        s = &body[close + quote.len_utf8()..];
    }
}

/// Resolve the five predefined entities plus numeric character references. An unknown
/// entity is an error: silently passing `&frac12;` through would put a literal `&frac12;`
/// in a trophy name.
fn decode_entities(s: &str) -> Result<String, TrophyError> {
    if !s.contains('&') {
        return Ok(s.to_string());
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let tail = &rest[amp..];
        let Some(semi) = tail.find(';') else {
            return Err(TrophyError::BadConf(format!("unterminated entity near {tail:?}")));
        };
        let name = &tail[1..semi];
        let ch = match name {
            "amp" => '&',
            "lt" => '<',
            "gt" => '>',
            "quot" => '"',
            "apos" => '\'',
            _ => {
                let code = if let Some(hex) = name.strip_prefix("#x").or_else(|| name.strip_prefix("#X")) {
                    u32::from_str_radix(hex, 16).ok()
                } else {
                    name.strip_prefix('#').and_then(|d| d.parse::<u32>().ok())
                };
                let Some(ch) = code.and_then(char::from_u32) else {
                    return Err(TrophyError::BadConf(format!("unknown entity &{name};")));
                };
                ch
            }
        };
        out.push(ch);
        rest = &tail[semi + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a TRP container over `(name, bytes)` files, so the reader is tested against
    /// the layout it documents rather than against a fixture nobody can ship.
    fn build_trp(files: &[(&str, &[u8])]) -> Vec<u8> {
        let header = TRP_HEADER_SIZE + files.len() * TRP_ENTRY_SIZE;
        let mut table = Vec::new();
        let mut body = Vec::new();
        for (name, bytes) in files {
            let mut entry = [0u8; TRP_ENTRY_SIZE];
            entry[..name.len()].copy_from_slice(name.as_bytes());
            entry[32..40].copy_from_slice(&((header + body.len()) as u64).to_be_bytes());
            entry[40..48].copy_from_slice(&(bytes.len() as u64).to_be_bytes());
            table.extend_from_slice(&entry);
            body.extend_from_slice(bytes);
        }
        let mut out = vec![0u8; TRP_HEADER_SIZE];
        out[0..4].copy_from_slice(&TRP_MAGIC.to_be_bytes());
        out[4..8].copy_from_slice(&TRP_VERSION.to_be_bytes());
        out[8..16].copy_from_slice(&((header + body.len()) as u64).to_be_bytes());
        out[0x10..0x14].copy_from_slice(&(files.len() as u32).to_be_bytes());
        out[0x14..0x18].copy_from_slice(&(TRP_ENTRY_SIZE as u32).to_be_bytes());
        out.extend_from_slice(&table);
        out.extend_from_slice(&body);
        out
    }

    const TROPCONF: &str = r#"<!--Sce-Np-Trophy-Signature: 00-->
<trophyconf version="1.1" platform="psp2" policy="small">
 <npcommid>NPWR00000_00</npcommid>
 <trophyset-version>01.01</trophyset-version>
 <parental-level license-area="default">0</parental-level>
 <group id="000"/>
 <trophy id="000" hidden="no" ttype="P" pid="-1"/>
 <trophy id="001" hidden="yes" ttype="G" pid="0"/>
 <trophy id="002" hidden="no" ttype="S" pid="0"/>
 <trophy id="003" hidden="no" ttype="B" pid="-1"/>
</trophyconf>
"#;

    const TROP_EN: &str = r#"<trophyconf version="1.1" platform="psp2" policy="small">
 <npcommid>NPWR00000_00</npcommid>
 <title-name>Set &amp; Match</title-name>
 <title-detail>A test set</title-detail>
 <group id="000">
  <name>First group</name>
  <detail>Group detail</detail>
 </group>
 <trophy id="000" hidden="no" ttype="P" pid="-1">
  <name>All of it</name>
  <detail>Unlock everything</detail>
 </trophy>
 <trophy id="001" hidden="yes" ttype="G" pid="0">
  <name>Secret</name>
  <detail>Hidden until earned</detail>
 </trophy>
 <trophy id="002" hidden="no" ttype="S" pid="0">
  <name>Halfway</name>
  <detail>Half of it</detail>
 </trophy>
 <trophy id="003" hidden="no" ttype="B" pid="-1">
  <name>First steps</name>
  <detail>Start playing</detail>
 </trophy>
</trophyconf>
"#;

    fn sample() -> Vec<u8> {
        build_trp(&[
            ("TROPCONF.SFM", TROPCONF.as_bytes()),
            ("TROP.SFM", TROP_EN.as_bytes()),
            ("ICON0.PNG", b"icon-bytes"),
            ("TROP002.PNG", b"trophy-2-icon"),
            ("GR000.PNG", b"group-0-icon"),
        ])
    }

    #[test]
    fn reads_container_entries() {
        let a = TrpArchive::parse(sample()).unwrap();
        assert_eq!(a.file("ICON0.PNG").unwrap(), b"icon-bytes");
        assert_eq!(a.file("icon0.png").unwrap(), b"icon-bytes");
        assert!(a.file("NOPE.PNG").is_none());
        assert_eq!(a.names().count(), 5);
    }

    #[test]
    fn rejects_a_non_trp() {
        assert_eq!(TrpArchive::parse(vec![0u8; 0x80]).unwrap_err(), TrophyError::NotTrp);
        assert!(matches!(
            TrpArchive::parse(vec![0xDC, 0xA2, 0x4D]).unwrap_err(),
            TrophyError::Truncated(_)
        ));
    }

    #[test]
    fn rejects_an_entry_running_past_the_end() {
        let mut bytes = sample();
        // Enlarge the first entry's size field beyond the buffer.
        let size_at = TRP_HEADER_SIZE + 40;
        bytes[size_at..size_at + 8].copy_from_slice(&u64::MAX.to_be_bytes());
        assert!(matches!(TrpArchive::parse(bytes).unwrap_err(), TrophyError::Truncated(_)));
    }

    #[test]
    fn parses_the_set_and_merges_localized_text() {
        let set = TrophySet::parse(sample(), 1).unwrap();
        assert_eq!(set.comm_id, "NPWR00000_00");
        assert_eq!(set.title, "Set & Match"); // entity decoded
        assert_eq!(set.detail, "A test set");
        assert_eq!(set.trophies.len(), 4);
        let t = set.trophy(1).unwrap();
        assert_eq!(t.grade, Grade::Gold);
        assert!(t.hidden);
        assert_eq!(t.group_id, 0);
        assert_eq!(t.name, "Secret");
        assert_eq!(t.detail, "Hidden until earned");
        assert!(!set.trophy(0).unwrap().hidden);
        assert_eq!(set.groups, vec![GroupDef {
            id: 0,
            name: "First group".into(),
            detail: "Group detail".into()
        }]);
    }

    #[test]
    fn counts_by_grade_over_the_set_and_one_group() {
        let set = TrophySet::parse(sample(), 1).unwrap();
        let all = set.counts(|_| true);
        assert_eq!(all, GradeCounts { total: 4, platinum: 1, gold: 1, silver: 1, bronze: 1 });
        let g0 = set.counts(|t| t.group_id == 0);
        assert_eq!(g0, GradeCounts { total: 2, platinum: 0, gold: 1, silver: 1, bronze: 0 });
    }

    #[test]
    fn serves_icons_by_id() {
        let set = TrophySet::parse(sample(), 1).unwrap();
        assert_eq!(set.game_icon().unwrap(), b"icon-bytes");
        assert_eq!(set.trophy_icon(2).unwrap(), b"trophy-2-icon");
        assert!(set.trophy_icon(0).is_none());
        assert_eq!(set.group_icon(0).unwrap(), b"group-0-icon");
        assert!(set.group_icon(-1).is_none());
    }

    #[test]
    fn prefers_the_conf_for_the_system_language() {
        let fr = TROP_EN.replace("All of it", "Tout");
        let bytes = build_trp(&[
            ("TROPCONF.SFM", TROPCONF.as_bytes()),
            ("TROP.SFM", TROP_EN.as_bytes()),
            ("TROP_02.SFM", fr.as_bytes()),
        ]);
        assert_eq!(TrophySet::parse(bytes.clone(), 2).unwrap().trophy(0).unwrap().name, "Tout");
        // No TROP_09.SFM in the container, so the default conf is used.
        assert_eq!(TrophySet::parse(bytes, 9).unwrap().trophy(0).unwrap().name, "All of it");
    }

    #[test]
    fn a_container_without_display_text_still_yields_the_set() {
        let bytes = build_trp(&[("TROPCONF.SFM", TROPCONF.as_bytes())]);
        let set = TrophySet::parse(bytes, 1).unwrap();
        assert_eq!(set.counts(|_| true).total, 4);
        assert_eq!(set.title, "");
        assert_eq!(set.trophy(0).unwrap().name, "");
    }

    #[test]
    fn a_conf_without_a_grade_is_an_error_not_a_guess() {
        let broken = TROPCONF.replace(r#"ttype="P""#, r#"ttype="X""#);
        let bytes = build_trp(&[("TROPCONF.SFM", broken.as_bytes())]);
        assert!(matches!(TrophySet::parse(bytes, 1).unwrap_err(), TrophyError::BadConf(_)));
    }

    #[test]
    fn a_container_without_a_conf_is_an_error() {
        let bytes = build_trp(&[("ICON0.PNG", b"x")]);
        assert_eq!(TrophySet::parse(bytes, 1).unwrap_err(), TrophyError::MissingConf);
    }

    /// The synthetic containers above test the reader against the layout it documents.
    /// This one tests it against a REAL retail TRP, which is the only thing that can catch
    /// the layout itself being wrong. Assertions are content-free (no name, count or title
    /// from the title is baked in) and the test skips when the private fixture is absent.
    #[test]
    fn reads_a_real_retail_trophy_set() {
        use crate::ingest::{testfix, vfs::DirVfs};
        let Some(dir) = testfix::game_dir() else { return };
        let Ok(game) = crate::ingest::pipeline::decrypt_container(&mut DirVfs::new(dir)) else {
            return;
        };
        // The set lives under a directory named for its own NP communication id.
        use crate::ingest::vfs::Vfs;
        let Some(path) =
            game.files.list().into_iter().find(|p| p.to_ascii_lowercase().ends_with("/trophy.trp"))
        else {
            return;
        };
        let dir_name = path.rsplit('/').nth(1).expect("TROPHY.TRP has a parent directory");
        let set = TrophySet::parse(game.file(&path).expect("read TROPHY.TRP"), 1)
            .expect("parse the title's own trophy set");

        assert_eq!(set.comm_id, dir_name, "conf npcommid disagrees with its directory");
        let counts = set.counts(|_| true);
        assert_eq!(counts.total as usize, set.trophies.len());
        assert_eq!(
            counts.total,
            counts.platinum + counts.gold + counts.silver + counts.bronze,
            "a trophy was counted with no grade"
        );
        assert!(counts.total > 0, "a shipped trophy set is never empty");
        assert!(set.game_icon().is_some(), "no ICON0.PNG in the container");
        assert_eq!(&set.game_icon().unwrap()[..4], b"\x89PNG", "ICON0.PNG is not a PNG");
        assert!(!set.title.is_empty(), "no localized <title-name>");
        for t in &set.trophies {
            assert!(!t.name.is_empty(), "trophy {} has no localized name", t.id);
            assert!(set.trophy_icon(t.id).is_some(), "trophy {} has no icon", t.id);
        }
    }

    #[test]
    fn unknown_entities_and_malformed_attributes_are_errors() {
        assert!(decode_entities("a &frac12; b").is_err());
        assert_eq!(decode_entities("&#65;&#x42;&amp;").unwrap(), "AB&");
        assert!(parse_attrs("id=000").is_err());
        assert!(parse_attrs(r#"id="000"#).is_err());
        assert_eq!(parse_attrs(r#" id="7" ttype='B' "#).unwrap(), vec![
            ("id", "7".to_string()),
            ("ttype", "B".to_string())
        ]);
    }
}
