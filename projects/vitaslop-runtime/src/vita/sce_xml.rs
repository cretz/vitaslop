//! `sce::Xml`: the Vita's C++ XML library, DOM half.
//!
//! # What this is and how its shape was established
//! `SceLibXml` is a C++ library, so the vitasdk NID database carries ITALIAN-MANGLED
//! names - `_ZNK3sce3Xml3Dom8Document13getFirstChildEy` - and a mangled name is a full
//! signature. Every prototype below is demangled from the NID's own name, so the argument
//! counts and types here are not RE guesses: `...getFirstChildEy` takes one `unsigned long
//! long`, `...parseEPKNS0_6StringEb` takes a `const String*` and a `bool`, and so on.
//!
//! What the names do NOT give is the object LAYOUTS, and those were read off the calling
//! title's own code (the sizes are in its allocations):
//!
//! ```text
//!   sce::Xml::MemAllocator            4 bytes  - a vptr; the title SUBCLASSES it and
//!                                                overwrites the pointer our ctor writes
//!   sce::Xml::Initializer             4 bytes  - embedded at +0x34 of a 0x38-byte object
//!   sce::Xml::InitParameter           8 bytes  - { MemAllocator* alloc; void* userData }
//!   sce::Xml::Dom::DocumentBuilder    4 bytes  - the title mallocs exactly 4 for it
//!   sce::Xml::String                  8 bytes  - { const char* p; SceSize n }, filled by
//!                                                the title INLINE (only the default
//!                                                constructor is an out-of-line symbol)
//! ```
//!
//! Every one of these is OPAQUE to the title: it allocates the storage and hands it
//! straight back to us. So the contents are ours to define, and each carries a magic word
//! - a handle we can validate - rather than anything the guest is expected to understand.
//! The only constraint the title imposes is the SIZE, which is why the sizes above are
//! evidence and the contents are not.
//!
//! # One global node arena, because `Node::Node(unsigned long long)` says so
//! `Node`'s only constructor takes a bare node id and `Node::getNodeName() const` takes
//! nothing but `this` - so a node id must identify its document all by itself. Ids are
//! therefore allocated from ONE arena across every document parsed in a run, and a
//! `Document` only has to name its root. Reading that constructor as `{document, index}`
//! would have made `Node::getNodeName` unimplementable, which is how the arena's shape was
//! settled rather than chosen.
//!
//! # Returning a `String`
//! AAPCS returns a composite larger than four bytes through a hidden pointer, so every
//! `String`-returning entry point takes the result buffer in `r0` and shifts its real
//! arguments up one - `getText(u64)` is `(r0 = out String*, r1 = this, r2:r3 = node)`.
//! A `u64` return, by contrast, is a fundamental type and comes back in `r0:r1`, which is
//! exactly what the calling title does with `getRoot` (`strd r0, r1, [sp, #0x10]`).
//!
//! The bytes a returned `String` points at have to live in GUEST memory for the title to
//! read them, and they must outlive the call. They are interned: each distinct string is
//! allocated once and handed back for every query that yields it, so a title walking a
//! document in a loop does not leak a copy per node.

use crate::host::{GuestCtx, Ptr, VitaState};
use crate::hostcall;

/// Magic stamped into each object we hand back, so a pointer that never came from the
/// matching constructor is caught instead of read as state.
const MAGIC_INITIALIZER: u32 = 0x584D_4C49; // "XMLI"
const MAGIC_BUILDER: u32 = 0x584D_4C42; // "XMLB"
const MAGIC_ALLOCATOR: u32 = 0x584D_4C41; // "XMLA"

/// `sce::Xml::Dom::NodeType`. **NOT the DOM level 1 numbering** - this was 1/2/3/9 here
/// until a calling title's own comparisons refuted it, and the refutation is worth keeping
/// because the failure was silent: every navigation call still worked, every guard that
/// asked "is this a node I may descend into" answered no, and the title read its whole
/// configuration as empty without one error anywhere.
///
/// The numbering below is read off the guard sites, which are the only oracle there is (no
/// public header carries the enum, and the NID database carries names, not values). A title
/// wraps every DOM walk in one of two guards:
///
/// ```text
///   getNodeType(n) == 4 || == 5   -> then getNodeName / getFirstChild   (CONTAINER)
///   getNodeType(n) == 8 || == 0x28 -> then getText                      (CHARACTER DATA)
/// ```
///
/// So there are two container types and two character-data types, and each guard names the
/// pair. `getRoot` hands back the document ELEMENT (see [`document_get_root`]) and that
/// value must pass the first guard, which pins ELEMENT to one of `4`/`5`; DOCUMENT takes
/// the other. Likewise TEXT takes one of `8`/`0x28` and CDATA the other - the guard would
/// not test two values if a CDATA section carried the same type as ordinary text.
///
/// Read as a bit set the four fall out as `4 = element`, `8 = text`, `+1 = is the document`,
/// `+0x20 = is a CDATA section`, and that reading is what settles which of each pair is
/// which. ATTRIBUTE is the one value here NOT pinned by any observation: nothing measured
/// asks a node's type before calling `getAttrName`/`getAttrValue`, so `2` is this bit
/// set's free slot rather than a measurement.
const NODE_TYPE_ELEMENT: u32 = 4;
const NODE_TYPE_ATTRIBUTE: u32 = 2;
const NODE_TYPE_TEXT: u32 = 8;
const NODE_TYPE_CDATA: u32 = 0x28;
const NODE_TYPE_DOCUMENT: u32 = 5;

/// What one node in the arena is. Attributes are nodes too, on their own sibling chain,
/// so `getFirstAttr`/`getSibling`/`getNodeName`/`getAttrValue` need no second code path.
#[derive(Default, Clone)]
pub struct XmlNode {
    pub kind: u32,
    pub name: String,
    /// An element's concatenated text, an attribute's value, a text node's content.
    pub value: String,
    pub parent: u64,
    pub first_child: u64,
    pub next_sibling: u64,
    pub first_attr: u64,
}

/// The whole `sce::Xml` state of a run: the node arena, the parsed documents' roots, and
/// the interned guest strings.
#[derive(Default)]
pub struct XmlState {
    /// Node id `n` is `nodes[n - 1]`; id 0 is "no node", which is what every navigation
    /// call returns at the end of a chain.
    pub nodes: Vec<XmlNode>,
    /// The document object each builder has produced, as `(builder id, root node)`.
    pub docs: Vec<(u32, u64)>,
    pub next_builder: u32,
    /// Interned guest copies of returned strings: content -> `(addr, len)`.
    pub interned: std::collections::HashMap<String, (u32, u32)>,
    /// Said once if a document fails to parse.
    reported_parse_error: bool,
}

impl XmlState {
    fn node(&self, id: u64) -> Option<&XmlNode> {
        if id == 0 {
            None
        } else {
            self.nodes.get((id - 1) as usize)
        }
    }

    fn push(&mut self, n: XmlNode) -> u64 {
        self.nodes.push(n);
        self.nodes.len() as u64
    }
}

// --- the parser ------------------------------------------------------------------
//
// A small, dependency-free XML reader: it exists to answer the DOM queries below, and the
// runtime compiles to wasm, so pulling in a general XML crate for it would cost far more
// than it buys. It handles what a title's data files contain - the declaration, comments,
// CDATA, elements, attributes, text and the five predefined entities - and REFUSES what it
// does not (a DTD, a processing instruction with a body it would have to interpret) rather
// than skipping it silently, because a document half-read is worse than one not read.

/// Expand the five predefined entities plus numeric character references. Anything else
/// starting `&` is left ALONE rather than dropped: an unknown entity in a title's own data
/// is more likely a literal ampersand it never escaped than a reference we should discard.
fn unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'&' {
            out.push(b[i] as char);
            i += 1;
            continue;
        }
        let Some(semi) = s[i..].find(';').map(|p| i + p) else {
            out.push('&');
            i += 1;
            continue;
        };
        let ent = &s[i + 1..semi];
        let replacement = match ent {
            "lt" => Some("<".to_string()),
            "gt" => Some(">".to_string()),
            "amp" => Some("&".to_string()),
            "quot" => Some("\"".to_string()),
            "apos" => Some("'".to_string()),
            _ => ent
                .strip_prefix('#')
                .and_then(|n| match n.strip_prefix('x').or_else(|| n.strip_prefix('X')) {
                    Some(hex) => u32::from_str_radix(hex, 16).ok(),
                    None => n.parse::<u32>().ok(),
                })
                .and_then(char::from_u32)
                .map(|c| c.to_string()),
        };
        match replacement {
            Some(r) => {
                out.push_str(&r);
                i = semi + 1;
            }
            None => {
                out.push('&');
                i += 1;
            }
        }
    }
    out
}

/// Parse `text` into `st`'s arena and return the DOCUMENT node's id, or `None` if the
/// document is not well-formed.
pub fn parse_document(st: &mut XmlState, text: &str) -> Option<u64> {
    let doc = st.push(XmlNode { kind: NODE_TYPE_DOCUMENT, ..Default::default() });
    let b = text.as_bytes();
    let mut i = 0usize;
    // The open-element stack, and each one's last child / last attribute, so appending is
    // O(1) instead of walking the sibling chain every time.
    let mut stack: Vec<(u64, u64, u64)> = vec![(doc, 0, 0)];
    while i < b.len() {
        if b[i] != b'<' {
            // Character data up to the next tag, attached to the enclosing element.
            let end = text[i..].find('<').map_or(b.len(), |p| i + p);
            let raw = &text[i..end];
            if !raw.trim().is_empty() {
                let content = unescape(raw);
                let (parent, last_child, _) = *stack.last()?;
                // Append to the element's own text as well as making a text node: a title
                // asks for either, and `getText` on an element must not depend on which.
                if let Some(n) = st.nodes.get_mut((parent - 1) as usize) {
                    n.value.push_str(&content);
                }
                let id = st.push(XmlNode {
                    kind: NODE_TYPE_TEXT,
                    value: content,
                    parent,
                    ..Default::default()
                });
                link_child(st, &mut stack, last_child, id);
            }
            i = end;
            continue;
        }
        if text[i..].starts_with("<!--") {
            i = text[i..].find("-->").map(|p| i + p + 3)?;
            continue;
        }
        if text[i..].starts_with("<![CDATA[") {
            let end = text[i..].find("]]>").map(|p| i + p)?;
            let content = text[i + 9..end].to_string();
            let (parent, last_child, _) = *stack.last()?;
            if let Some(n) = st.nodes.get_mut((parent - 1) as usize) {
                n.value.push_str(&content);
            }
            // A CDATA section is character data with its own type, not a text node: the
            // guard sites test `8 || 0x28` precisely because the two are told apart.
            let id =
                st.push(XmlNode { kind: NODE_TYPE_CDATA, value: content, parent, ..Default::default() });
            link_child(st, &mut stack, last_child, id);
            i = end + 3;
            continue;
        }
        if text[i..].starts_with("<?") {
            // The XML declaration, or a processing instruction. Neither contributes a node.
            i = text[i..].find("?>").map(|p| i + p + 2)?;
            continue;
        }
        if text[i..].starts_with("<!") {
            // A DOCTYPE or other declaration. Skipping it wholesale is right only when it
            // carries no internal subset - one that does could define entities this parser
            // would then fail to expand, so a `[` here is a refusal, not a skip.
            let end = text[i..].find('>').map(|p| i + p)?;
            if text[i..end].contains('[') {
                return None;
            }
            i = end + 1;
            continue;
        }
        if text[i..].starts_with("</") {
            let end = text[i..].find('>').map(|p| i + p)?;
            let name = text[i + 2..end].trim();
            let (open, _, _) = *stack.last()?;
            // A mismatched close tag is not well-formed. Refusing beats carrying on with a
            // tree whose shape no longer matches the document.
            if st.node(open)?.name != name || stack.len() < 2 {
                return None;
            }
            stack.pop();
            i = end + 1;
            continue;
        }
        // An element. Find its `>`, honouring quoted attribute values so a `>` inside one
        // does not end the tag early.
        let (tag_end, self_closing) = scan_tag(text, i)?;
        let inner = &text[i + 1..tag_end];
        let inner = inner.strip_suffix('/').unwrap_or(inner);
        let mut parts = inner.splitn(2, |c: char| c.is_ascii_whitespace());
        let name = parts.next()?.trim();
        if name.is_empty() {
            return None;
        }
        let (parent, last_child, _) = *stack.last()?;
        let id = st.push(XmlNode {
            kind: NODE_TYPE_ELEMENT,
            name: name.to_string(),
            parent,
            ..Default::default()
        });
        link_child(st, &mut stack, last_child, id);
        let mut last_attr = 0u64;
        for (k, v) in parse_attrs(parts.next().unwrap_or("")) {
            let a = st.push(XmlNode {
                kind: NODE_TYPE_ATTRIBUTE,
                name: k,
                value: v,
                parent: id,
                ..Default::default()
            });
            if last_attr == 0 {
                st.nodes[(id - 1) as usize].first_attr = a;
            } else {
                st.nodes[(last_attr - 1) as usize].next_sibling = a;
            }
            last_attr = a;
        }
        if !self_closing {
            stack.push((id, 0, last_attr));
        }
        i = tag_end + 1;
    }
    // Every element that was opened must have been closed.
    if stack.len() != 1 {
        return None;
    }
    Some(doc)
}

/// Append `id` as the next child of the innermost open element, given that element's
/// previous last child.
fn link_child(st: &mut XmlState, stack: &mut [(u64, u64, u64)], last_child: u64, id: u64) {
    let Some(top) = stack.last_mut() else { return };
    if last_child == 0 {
        st.nodes[(top.0 - 1) as usize].first_child = id;
    } else {
        st.nodes[(last_child - 1) as usize].next_sibling = id;
    }
    top.1 = id;
}

/// Find the `>` that ends the tag starting at `open`, and whether the tag self-closes.
/// Quoted attribute values are skipped over, so `<a b="x>y"/>` ends at the right place.
fn scan_tag(text: &str, open: usize) -> Option<(usize, bool)> {
    let b = text.as_bytes();
    let mut i = open + 1;
    let mut quote = 0u8;
    while i < b.len() {
        let c = b[i];
        if quote != 0 {
            if c == quote {
                quote = 0;
            }
        } else if c == b'"' || c == b'\'' {
            quote = c;
        } else if c == b'>' {
            let self_closing = i > open + 1 && b[i - 1] == b'/';
            return Some((i, self_closing));
        }
        i += 1;
    }
    None
}

/// `name="value"` pairs from the inside of a start tag, in document order.
fn parse_attrs(s: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let b = s.as_bytes();
    let mut i = 0usize;
    while i < b.len() {
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        let start = i;
        while i < b.len() && b[i] != b'=' && !b[i].is_ascii_whitespace() {
            i += 1;
        }
        if i == start {
            break;
        }
        let name = s[start..i].to_string();
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= b.len() || b[i] != b'=' {
            // A bare attribute with no value. XML forbids it, but recording it with an
            // empty value beats dropping the rest of the tag on the floor.
            out.push((name, String::new()));
            continue;
        }
        i += 1;
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= b.len() {
            out.push((name, String::new()));
            break;
        }
        let quote = b[i];
        let (value, next) = if quote == b'"' || quote == b'\'' {
            let end = s[i + 1..].find(quote as char).map_or(b.len(), |p| i + 1 + p);
            (&s[i + 1..end.min(s.len())], end + 1)
        } else {
            let end = s[i..].find(|c: char| c.is_ascii_whitespace()).map_or(b.len(), |p| i + p);
            (&s[i..end], end)
        };
        out.push((name, unescape(value)));
        i = next;
    }
    out
}

// --- the guest-facing entry points -------------------------------------------------

/// Copy `s` into guest memory (NUL-terminated, so a title that treats the pointer as a C
/// string also works) and return `(addr, len)`. Interned: the same content is allocated
/// once per run, so walking a document does not leak a copy per query.
fn intern(ctx: &mut GuestCtx, st: &mut VitaState, s: &str) -> (u32, u32) {
    if let Some(&hit) = st.xml.interned.get(s) {
        return hit;
    }
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    let addr = st.galloc(bytes.len() as u32, 4);
    ctx.write_bytes(addr, &bytes);
    let entry = (addr, s.len() as u32);
    st.xml.interned.insert(s.to_string(), entry);
    entry
}

/// Write an `sce::Xml::String` - `{ const char* p; SceSize n }` - through an out-pointer.
fn write_string(ctx: &mut GuestCtx, out: u32, addr: u32, len: u32) {
    if out != 0 {
        ctx.write_u32(out, addr);
        ctx.write_u32(out + 4, len);
    }
}

/// Read one, the same way.
fn read_string(ctx: &GuestCtx, p: u32) -> String {
    if p == 0 {
        return String::new();
    }
    let (addr, len) = (ctx.read_u32(p), ctx.read_u32(p + 4));
    if addr == 0 || len == 0 {
        return String::new();
    }
    let bytes = ctx.read_bytes(addr, len as usize);
    String::from_utf8_lossy(&bytes).into_owned()
}

/// `sce::Xml::MemAllocator::MemAllocator()`
///
/// The base class's constructor writes its vptr. The calling title immediately overwrites
/// that word with its OWN vtable (it subclasses the allocator to route allocations through
/// its heap), so what goes here is never dispatched through - it only has to be a value
/// the object can be told apart by if it is ever handed back before the overwrite.
///
/// The allocator itself is never CALLED, because every allocation this library makes is
/// host-side: the document lives in the arena above, not in guest memory. That is a real
/// difference from hardware - a title watching its own heap will not see the parse - and it
/// is the reason the parse cannot fail for lack of memory here.
#[hostcall]
pub(super) fn mem_allocator_ctor(ctx: &mut GuestCtx, _st: &mut VitaState, this: Ptr) -> u32 {
    if !this.is_null() {
        ctx.write_u32(this.addr(), MAGIC_ALLOCATOR);
    }
    this.addr()
}

/// `sce::Xml::MemAllocator::~MemAllocator()`
#[hostcall]
pub(super) fn mem_allocator_dtor(_ctx: &mut GuestCtx, _st: &mut VitaState, this: Ptr) -> u32 {
    this.addr()
}

/// `sce::Xml::Initializer::Initializer()`
#[hostcall]
pub(super) fn initializer_ctor(ctx: &mut GuestCtx, _st: &mut VitaState, this: Ptr) -> u32 {
    if !this.is_null() {
        ctx.write_u32(this.addr(), MAGIC_INITIALIZER);
    }
    this.addr()
}

/// `sce::Xml::Initializer::~Initializer()`
#[hostcall]
pub(super) fn initializer_dtor(_ctx: &mut GuestCtx, _st: &mut VitaState, this: Ptr) -> u32 {
    this.addr()
}

/// `sce::Xml::Initializer::initialize(const InitParameter*)`
///
/// `InitParameter` is `{ MemAllocator* alloc; void* userData }`. Neither is consumed - see
/// [`mem_allocator_ctor`] for why there is nothing for the allocator to allocate - so this
/// only confirms the initializer is one of ours. Returns 0 for success, as the whole
/// library's `SceInt32` convention does.
#[hostcall]
pub(super) fn initializer_initialize(ctx: &mut GuestCtx, _st: &mut VitaState, this: Ptr, _param: Ptr) -> i32 {
    if this.is_null() || ctx.read_u32(this.addr()) != MAGIC_INITIALIZER {
        -1
    } else {
        0
    }
}

/// `sce::Xml::Dom::DocumentBuilder::DocumentBuilder()`
///
/// The builder object is FOUR BYTES - the calling title mallocs exactly that for it - so
/// all it can hold is an id into the host-side table, which is what it holds.
#[hostcall]
pub(super) fn builder_ctor(ctx: &mut GuestCtx, st: &mut VitaState, this: Ptr) -> u32 {
    if !this.is_null() {
        st.xml.next_builder += 1;
        let id = st.xml.next_builder;
        st.xml.docs.push((id, 0));
        // The magic shares the word with the id: the id is small and the tag is the high
        // half, so a pointer that never came from here fails the tag check.
        ctx.write_u32(this.addr(), (MAGIC_BUILDER & 0xFFFF_0000) | (id & 0xFFFF));
    }
    this.addr()
}

/// The builder id in a `DocumentBuilder`, or `None` if this is not one of ours.
fn builder_id(ctx: &GuestCtx, this: u32) -> Option<u32> {
    if this == 0 {
        return None;
    }
    let w = ctx.read_u32(this);
    if w & 0xFFFF_0000 != (MAGIC_BUILDER & 0xFFFF_0000) {
        return None;
    }
    Some(w & 0xFFFF)
}

/// `sce::Xml::Dom::DocumentBuilder::~DocumentBuilder()`
#[hostcall]
pub(super) fn builder_dtor(_ctx: &mut GuestCtx, _st: &mut VitaState, this: Ptr) -> u32 {
    // The parsed nodes deliberately OUTLIVE the builder: the title keeps `Document` and
    // `Node` values (both are just ids) and goes on querying them after the builder is
    // gone, which is exactly what the DOM contract allows.
    this.addr()
}

/// `sce::Xml::Dom::DocumentBuilder::initialize(const Initializer*)`
#[hostcall]
pub(super) fn builder_initialize(ctx: &mut GuestCtx, _st: &mut VitaState, this: Ptr, _init: Ptr) -> i32 {
    if builder_id(ctx, this.addr()).is_some() {
        0
    } else {
        -1
    }
}

/// `setResolveEntity(bool)` / `setSkipIgnorableText(bool)` / `setSkipIgnorableWhiteSpace(bool)`
///
/// All three are accepted and none changes what this parser does, which is worth being
/// precise about rather than leaving implied:
/// - entity resolution is ALWAYS on here (see [`unescape`]), so `setResolveEntity(true)` -
///   which is what the calling title asks for - is already the behaviour;
/// - ignorable text and whitespace are ALWAYS skipped: a run of characters that is
///   entirely whitespace produces no text node at all.
///
/// A title that asked for the opposite of either would get a document with fewer text
/// nodes than it expects. Nothing observed asks for that, and inventing the other mode
/// unexercised would be worse than this note.
#[hostcall]
pub(super) fn builder_set_flag(ctx: &mut GuestCtx, _st: &mut VitaState, this: Ptr, _on: bool) -> i32 {
    if builder_id(ctx, this.addr()).is_some() {
        0
    } else {
        -1
    }
}

/// `sce::Xml::Dom::DocumentBuilder::parse(const String*, bool)`
///
/// The document text is `String`'s `{p, n}`, read straight out of guest memory. A negative
/// return is the failure the caller tests for (`lsr #31; eor #1` at the call site - it
/// keeps only the sign bit), so a malformed document reports one rather than handing back
/// an empty tree that would read as a valid, childless document.
#[hostcall]
pub(super) fn builder_parse(ctx: &mut GuestCtx, st: &mut VitaState, this: Ptr, xml: Ptr, _b: bool) -> i32 {
    do_parse(ctx, st, this.addr(), xml.addr())
}

/// `VITASLOP_XML_DUMP=<dir>`: write every document handed to `parse` into `<dir>` as
/// `doc-<n>.xml`, numbered in parse order.
///
/// A title's XML is not on disk in readable form - it arrives through a resource manager
/// that decompresses an archive record - so the only place the document exists as text is
/// the moment it reaches this call. Without it, "the title read its configuration as X"
/// cannot be checked against what the configuration actually SAYS, and a value that came
/// out wrong is indistinguishable from a value that was wrong in the file.
fn dump_document(text: &str) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    let Ok(dir) = std::env::var("VITASLOP_XML_DUMP") else { return };
    let n = N.fetch_add(1, Ordering::Relaxed);
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(std::path::Path::new(&dir).join(format!("doc-{n}.xml")), text);
}

/// The body of [`builder_parse`], as a plain function so it can use early returns.
fn do_parse(ctx: &mut GuestCtx, st: &mut VitaState, this: u32, xml: u32) -> i32 {
    let Some(id) = builder_id(ctx, this) else { return -1 };
    let text = read_string(ctx, xml);
    if text.is_empty() {
        return -1;
    }
    dump_document(&text);
    match parse_document(&mut st.xml, &text) {
        Some(root) => {
            if let Some(e) = st.xml.docs.iter_mut().find(|(b, _)| *b == id) {
                e.1 = root;
            }
            0
        }
        None => {
            if !st.xml.reported_parse_error {
                st.xml.reported_parse_error = true;
                tracing::warn!(
                    target: "vitaslop::cb",
                    bytes = text.len(),
                    head = %text.chars().take(80).collect::<String>(),
                    "sce::Xml::Dom::DocumentBuilder::parse: the document is not well-formed by \
                     this parser. The title is told so (a negative result), which is the same \
                     thing hardware would say about a broken document - but if the document is \
                     in fact fine, the gap is in the parser and this line names it."
                );
            }
            -1
        }
    }
}

/// `sce::Xml::Dom::DocumentBuilder::getDocument()`
///
/// Returns a `Document` BY VALUE, so `r0` is the hidden result pointer and the builder is
/// in `r1`. A `Document` is opaque and only ever handed back to us, so it holds the one
/// thing every `Document` method needs: its root node id.
pub(super) fn builder_get_document(ctx: &mut GuestCtx, st: &mut VitaState) {
    let (out, this) = (ctx.arg(0), ctx.arg(1));
    let root = builder_id(ctx, this)
        .and_then(|id| st.xml.docs.iter().find(|(b, _)| *b == id).map(|&(_, r)| r))
        .unwrap_or(0);
    if out != 0 {
        ctx.write_u32(out, root as u32);
        ctx.write_u32(out + 4, (root >> 32) as u32);
    }
    ctx.ret(out);
}

/// The root node id held in a `Document`.
fn doc_root(ctx: &GuestCtx, this: u32) -> u64 {
    if this == 0 {
        return 0;
    }
    u64::from(ctx.read_u32(this)) | (u64::from(ctx.read_u32(this + 4)) << 32)
}

/// `sce::Xml::Dom::Document::~Document()`
#[hostcall]
pub(super) fn document_dtor(_ctx: &mut GuestCtx, _st: &mut VitaState, this: Ptr) -> u32 {
    this.addr()
}

/// `sce::Xml::Dom::Document::getRoot() const`
///
/// The DOCUMENT ELEMENT - the single outermost element - not the document node itself,
/// which is what a title expects to start walking from.
pub(super) fn document_get_root(ctx: &mut GuestCtx, st: &mut VitaState) {
    let doc = doc_root(ctx, ctx.arg(0));
    let root = st.xml.node(doc).map_or(0, |n| n.first_child);
    ret_u64(ctx, root);
}

/// Return a `u64` in `r0:r1` - a fundamental type, so no hidden pointer is involved.
/// The same pair `sceKernelGetProcessTimeWide` writes.
fn ret_u64(ctx: &mut GuestCtx, v: u64) {
    ctx.regs[0] = v as u32;
    ctx.regs[1] = (v >> 32) as u32;
}

/// The `u64` node argument of a `Document::getX(unsigned long long) const`. It is an
/// 8-byte type after the `this` pointer, so AAPCS aligns it to an even register pair -
/// `r1` is SKIPPED and the id is in `r2:r3`. Reading it from `r1:r2` would give a node id
/// with half of `this` in it.
fn node_arg(ctx: &GuestCtx) -> u64 {
    u64::from(ctx.arg(2)) | (u64::from(ctx.arg(3)) << 32)
}

/// `sce::Xml::Dom::Document::getFirstChild(unsigned long long) const`
pub(super) fn document_get_first_child(ctx: &mut GuestCtx, st: &mut VitaState) {
    let id = node_arg(ctx);
    let child = st.xml.node(id).map_or(0, |n| n.first_child);
    ret_u64(ctx, child);
}

/// `sce::Xml::Dom::Document::getSibling(unsigned long long) const`
pub(super) fn document_get_sibling(ctx: &mut GuestCtx, st: &mut VitaState) {
    let id = node_arg(ctx);
    let sib = st.xml.node(id).map_or(0, |n| n.next_sibling);
    ret_u64(ctx, sib);
}

/// `sce::Xml::Dom::Document::getFirstAttr(unsigned long long) const`
pub(super) fn document_get_first_attr(ctx: &mut GuestCtx, st: &mut VitaState) {
    let id = node_arg(ctx);
    let attr = st.xml.node(id).map_or(0, |n| n.first_attr);
    ret_u64(ctx, attr);
}

/// `sce::Xml::Dom::Document::getNodeType(unsigned long long) const`
pub(super) fn document_get_node_type(ctx: &mut GuestCtx, st: &mut VitaState) {
    let id = node_arg(ctx);
    let kind = st.xml.node(id).map_or(0, |n| n.kind);
    ctx.ret(kind);
}

/// `sce::Xml::Dom::Document::getNodeName(unsigned long long) const`
///
/// Returns a `String`, so `r0` is the hidden result pointer and the arguments shift up:
/// `r1` is `this` and the node id lands in `r2:r3` (no alignment padding is needed - `r2`
/// is already even).
pub(super) fn document_get_node_name(ctx: &mut GuestCtx, st: &mut VitaState) {
    let (out, id) = (ctx.arg(0), node_arg(ctx));
    let name = st.xml.node(id).map(|n| n.name.clone()).unwrap_or_default();
    let (addr, len) = intern(ctx, st, &name);
    write_string(ctx, out, addr, len);
    ctx.ret(out);
}

/// `sce::Xml::Dom::Document::getText(unsigned long long) const`
///
/// An element's character data, an attribute's value, a text node's content - the one
/// query that answers "what does this node say", whichever kind it is.
pub(super) fn document_get_text(ctx: &mut GuestCtx, st: &mut VitaState) {
    let (out, id) = (ctx.arg(0), node_arg(ctx));
    let text = st.xml.node(id).map(|n| n.value.clone()).unwrap_or_default();
    let (addr, len) = intern(ctx, st, &text);
    write_string(ctx, out, addr, len);
    ctx.ret(out);
}

/// `sce::Xml::Dom::Node::Node(unsigned long long)`
///
/// A `Node` is the id and nothing else - see the module header for why the id has to
/// identify its document by itself. The `u64` is the FIRST argument after `this`, so it
/// takes the aligned pair `r2:r3`.
pub(super) fn node_ctor(ctx: &mut GuestCtx, _st: &mut VitaState) {
    let (this, id) = (ctx.arg(0), node_arg(ctx));
    if this != 0 {
        ctx.write_u32(this, id as u32);
        ctx.write_u32(this + 4, (id >> 32) as u32);
    }
    ctx.ret(this);
}

/// `sce::Xml::Dom::Node::~Node()`
#[hostcall]
pub(super) fn node_dtor(_ctx: &mut GuestCtx, _st: &mut VitaState, this: Ptr) -> u32 {
    this.addr()
}

/// `sce::Xml::Dom::Node::getNodeName() const` - `r0` is the hidden `String*`, `r1` is the
/// `Node*`.
pub(super) fn node_get_node_name(ctx: &mut GuestCtx, st: &mut VitaState) {
    let (out, this) = (ctx.arg(0), ctx.arg(1));
    let id = doc_root(ctx, this); // a Node holds its id in the same two words
    let name = st.xml.node(id).map(|n| n.name.clone()).unwrap_or_default();
    let (addr, len) = intern(ctx, st, &name);
    write_string(ctx, out, addr, len);
    ctx.ret(out);
}

/// `sce::Xml::Dom::Node::getNodeValue() const`
pub(super) fn node_get_node_value(ctx: &mut GuestCtx, st: &mut VitaState) {
    let (out, this) = (ctx.arg(0), ctx.arg(1));
    let id = doc_root(ctx, this);
    let value = st.xml.node(id).map(|n| n.value.clone()).unwrap_or_default();
    let (addr, len) = intern(ctx, st, &value);
    write_string(ctx, out, addr, len);
    ctx.ret(out);
}

/// `sce::Xml::String::String()` - the empty string.
#[hostcall]
pub(super) fn string_ctor(ctx: &mut GuestCtx, _st: &mut VitaState, this: Ptr) -> u32 {
    if !this.is_null() {
        ctx.write_u32(this.addr(), 0);
        ctx.write_u32(this.addr() + 4, 0);
    }
    this.addr()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walk(st: &XmlState, id: u64) -> Vec<(u32, String, String)> {
        let mut out = Vec::new();
        let mut cur = st.node(id).map_or(0, |n| n.first_child);
        while cur != 0 {
            let n = st.node(cur).unwrap();
            out.push((n.kind, n.name.clone(), n.value.clone()));
            cur = n.next_sibling;
        }
        out
    }

    #[test]
    fn parses_elements_attributes_and_text() {
        let mut st = XmlState::default();
        let doc = parse_document(
            &mut st,
            r#"<?xml version="1.0"?>
               <!-- a comment -->
               <root a="1" b="two &amp; more">
                 <kid>hello</kid>
                 <kid/>
                 <![CDATA[<raw>]]>
               </root>"#,
        )
        .expect("well formed");
        let root = st.node(doc).unwrap().first_child;
        assert_eq!(st.node(root).unwrap().name, "root");
        // Attributes, in document order, with entities expanded.
        let mut attrs = Vec::new();
        let mut a = st.node(root).unwrap().first_attr;
        while a != 0 {
            let n = st.node(a).unwrap();
            attrs.push((n.name.clone(), n.value.clone()));
            a = n.next_sibling;
        }
        assert_eq!(
            attrs,
            vec![("a".into(), "1".into()), ("b".into(), "two & more".into())]
        );
        let kids = walk(&st, root);
        assert_eq!(kids[0].1, "kid");
        assert_eq!(st.node(st.node(root).unwrap().first_child).unwrap().value, "hello");
        // The self-closing element is a real, childless node, and the CDATA is character
        // data of its own type.
        assert_eq!(kids.iter().filter(|(k, _, _)| *k == NODE_TYPE_ELEMENT).count(), 2);
        assert!(kids.iter().any(|(k, _, v)| *k == NODE_TYPE_CDATA && v == "<raw>"));
    }

    /// The node types a calling title's guards test for, pinned as VALUES rather than as
    /// names. This is the assertion that was missing when the DOM level 1 numbering was
    /// here: every structural test still passed under 1/2/3/9, because the tree is the
    /// same shape either way and only the guest ever compares the numbers.
    ///
    /// A title guards a descend with `type == 4 || type == 5` and a `getText` with
    /// `type == 8 || type == 0x28`, so a document element must answer the first pair and
    /// text and CDATA must each answer the second.
    #[test]
    fn node_types_answer_the_guards_a_title_writes() {
        let mut st = XmlState::default();
        let doc = parse_document(&mut st, r#"<Environment><Course>7</Course><![CDATA[x]]></Environment>"#)
            .expect("well formed");
        let container = |k: u32| k == 4 || k == 5;
        let chardata = |k: u32| k == 8 || k == 0x28;
        assert!(container(st.node(doc).unwrap().kind), "the document node");
        let root = st.node(doc).unwrap().first_child;
        assert!(container(st.node(root).unwrap().kind), "the document element");
        let course = st.node(root).unwrap().first_child;
        assert!(container(st.node(course).unwrap().kind), "a nested element");
        assert!(chardata(st.node(st.node(course).unwrap().first_child).unwrap().kind), "text");
        let cdata = st.node(course).unwrap().next_sibling;
        assert!(chardata(st.node(cdata).unwrap().kind), "a CDATA section");
        assert_ne!(st.node(cdata).unwrap().kind, st.node(st.node(course).unwrap().first_child).unwrap().kind);
    }

    /// A `>` inside a quoted attribute value must not end the tag.
    #[test]
    fn quoted_gt_does_not_end_a_tag() {
        let mut st = XmlState::default();
        let doc = parse_document(&mut st, r#"<a b="x>y"><c/></a>"#).expect("well formed");
        let root = st.node(doc).unwrap().first_child;
        assert_eq!(st.node(root).unwrap().name, "a");
        assert_eq!(st.node(st.node(root).unwrap().first_attr).unwrap().value, "x>y");
        assert_eq!(st.node(st.node(root).unwrap().first_child).unwrap().name, "c");
    }

    /// A mismatched or unclosed tag is REFUSED, not silently accepted with a wrong shape.
    #[test]
    fn malformed_is_refused() {
        let mut st = XmlState::default();
        assert!(parse_document(&mut st, "<a><b></a></b>").is_none());
        assert!(parse_document(&mut XmlState::default(), "<a>").is_none());
    }
}
