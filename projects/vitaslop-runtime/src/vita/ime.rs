//! SceIme (libime): the system on-screen keyboard a game opens over its own picture, typed on
//! here from the host's keyboard.
//!
//! # The event model
//! The game opens the IME with a handler and pumps `sceImeUpdate` once a frame; every event is
//! delivered by calling that handler, on the pumping thread, from inside `sceImeUpdate`. The
//! library keeps the text: before an `UPDATE_TEXT` it writes the current text (UTF-16, NUL
//! terminated) into the game's `inputTextBuffer`, which is where the game reads it. The game
//! closes the IME itself, typically on `PRESS_ENTER` or `PRESS_CLOSE`.
//!
//! # Keys
//! Each `sceImeUpdate` takes the keys typed since the last one ([`crate::world::World::poll_text`]):
//! a character is inserted at the caret (through the game's text filter when it gave one), Backspace
//! deletes before the caret, Left/Right move it, Enter is `PRESS_ENTER` (a newline with the
//! MULTILINE option), Escape is `PRESS_CLOSE`.
//!
//! # Guest calls and the replay contract
//! The filter and the handler are guest functions, so `sceImeUpdate` is re-run once per guest call
//! it makes (see `VitaState::call_guest`). Everything it changes is therefore staged: the keys are
//! polled once and kept, the events are built from them (with the filter's answers) on a copy of
//! the text and committed once all of them are known, and the delivery walks a stored outbox.

use crate::host::{GuestCall, GuestCtx, Ptr, VitaState};
use crate::hostcall;
use crate::world::TextKey;

const ERROR_ALREADY_OPENED: i32 = 0x8010_0700u32 as i32;
const ERROR_NOT_OPENED: i32 = 0x8010_0701u32 as i32;
const ERROR_INVALID_POINTER: i32 = 0x8010_0702u32 as i32;
const ERROR_INVALID_PARAM: i32 = 0x8010_0703u32 as i32;
const ERROR_INVALID_TEXT: i32 = 0x8010_0706u32 as i32;
const ERROR_INVALID_INPUT_METHOD: i32 = 0x8010_0712u32 as i32;
const ERROR_INVALID_SUPPORTED_LANGUAGES: i32 = 0x8010_0713u32 as i32;
const ERROR_INVALID_TYPE: i32 = 0x8010_0714u32 as i32;
const ERROR_INVALID_OPTION: i32 = 0x8010_0715u32 as i32;
const ERROR_INVALID_WORK: i32 = 0x8010_0716u32 as i32;
const ERROR_INVALID_HANDLER: i32 = 0x8010_0718u32 as i32;
const ERROR_INVALID_MAX_TEXT_LENGTH: i32 = 0x8010_0719u32 as i32;
const ERROR_INVALID_INPUT_TEXT_BUFFER: i32 = 0x8010_071Au32 as i32;
const ERROR_INVALID_RESERVED: i32 = 0x8010_071Bu32 as i32;
const ERROR_INVALID_ENTER_LABEL: i32 = 0x8010_071Cu32 as i32;

/// `SceImeEventId`.
const EVENT_OPEN: u32 = 0;
const EVENT_UPDATE_TEXT: u32 = 1;
const EVENT_UPDATE_CARET: u32 = 2;
const EVENT_PRESS_CLOSE: u32 = 4;
const EVENT_PRESS_ENTER: u32 = 5;

/// `SceImeParam` offsets.
mod param {
    pub const INPUT_METHOD: u32 = 0x04;
    pub const SUPPORTED_LANGUAGES: u32 = 0x08;
    pub const TYPE: u32 = 0x14;
    pub const OPTION: u32 = 0x18;
    pub const WORK: u32 = 0x1C;
    pub const ARG: u32 = 0x20;
    pub const HANDLER: u32 = 0x24;
    pub const FILTER: u32 = 0x28;
    pub const INITIAL_TEXT: u32 = 0x2C;
    pub const MAX_TEXT_LENGTH: u32 = 0x30;
    pub const INPUT_TEXT_BUFFER: u32 = 0x34;
    pub const ENTER_LABEL: u32 = 0x38;
    pub const RESERVED: u32 = 0x39;
}

const TYPE_MAX: u32 = 5;
const ENTER_LABEL_MAX: u8 = 3;
/// `SCE_IME_OPTION_MULTILINE | NO_AUTO_CAPITALIZATION | NO_ASSISTANCE`.
const OPTION_MASK: u32 = 0x7;
const OPTION_MULTILINE: u32 = 0x1;
/// The twenty `SCE_IME_LANGUAGE_*` bits.
const LANGUAGE_MASK: u64 = 0x000F_FFFF;
const MAX_TEXT_LENGTH: u32 = 2048;
/// `SCE_IME_WORK_BUFFER_SIZE`: the library's own memory, which the event record and the filter's
/// buffers live in here.
const WORK_SIZE: u32 = 20 * 1024;
/// Where in the work buffer the event record is built, and the filter's in/out buffers.
const WORK_EVENT: u32 = 0;
const WORK_FILTER_OUT_LEN: u32 = 0x40;
const WORK_FILTER_SRC: u32 = 0x80;
const WORK_FILTER_OUT: u32 = 0x100;
/// How many UTF-16 units one key can become through a filter.
const FILTER_OUT_MAX: usize = 0x80;

/// The on-screen keyboard panel the OPEN event reports, docked across the bottom of the 960x544
/// screen. This front end draws no panel (the host keyboard types), so this is only the extent
/// the game is told to keep clear.
const PANEL: (u32, u32, u32, u32) = (0, 544 - PANEL_HEIGHT, 960, PANEL_HEIGHT);
const PANEL_HEIGHT: u32 = 272;

/// One event waiting to be handed to the game: its id, the ten words of its payload, and - for
/// a text update - the text to write into `inputTextBuffer` first.
#[derive(Clone)]
struct Event {
    id: u32,
    data: [u32; 10],
    text: Option<Vec<u16>>,
}

struct Session {
    handler: u32,
    arg: u32,
    filter: u32,
    work: u32,
    input_buf: u32,
    max_len: usize,
    multiline: bool,
    text: Vec<u16>,
    caret: usize,
    open_pending: bool,
    preedit: (u32, u32, u32),
}

#[derive(Default)]
pub struct ImeState {
    session: Option<Session>,
    /// Keys polled for the update in progress (kept across its replays).
    keys: Option<Vec<TextKey>>,
    /// Events built for the update in progress, being delivered.
    outbox: Option<Vec<Event>>,
    /// The event being delivered when the handler called `sceImeClose`: no later event goes out.
    closed_during: Option<usize>,
    /// The handler, its argument and the work buffer of a session the handler closed mid-delivery.
    closed_target: Option<(u32, u32, u32)>,
    delivering: usize,
}

impl ImeState {
    /// Whether a game has the on-screen keyboard open (a front end routes typing to it).
    pub fn is_open(&self) -> bool {
        self.session.is_some()
    }
}

fn read_text(ctx: &GuestCtx, addr: u32, max: usize) -> Vec<u16> {
    let mut out = Vec::new();
    while out.len() < max {
        let u = ctx.read_u16(addr + 2 * out.len() as u32);
        if u == 0 {
            break;
        }
        out.push(u);
    }
    out
}

fn write_text(ctx: &mut GuestCtx, addr: u32, text: &[u16]) {
    let mut bytes: Vec<u8> = text.iter().flat_map(|u| u.to_le_bytes()).collect();
    bytes.extend_from_slice(&[0, 0]);
    ctx.write_bytes(addr, &bytes);
}

/// SceInt32 sceImeOpen(const SceImeParam *param)
#[hostcall]
pub(super) fn open(ctx: &mut GuestCtx, st: &mut VitaState, p: Ptr) -> i32 {
    open_in(ctx, st, p)
}

fn open_in(ctx: &mut GuestCtx, st: &mut VitaState, p: Ptr) -> i32 {
    if st.ime.session.is_some() {
        return ERROR_ALREADY_OPENED;
    }
    if p.is_null() {
        return ERROR_INVALID_POINTER;
    }
    let a = p.addr();
    let word = |off: u32| ctx.read_u32(a + off);
    let languages = u64::from(word(param::SUPPORTED_LANGUAGES))
        | (u64::from(word(param::SUPPORTED_LANGUAGES + 4)) << 32);
    let enter_label = ctx.read_bytes(a + param::ENTER_LABEL, 1)[0];
    let (option, max_len, buf, work, handler) = (
        word(param::OPTION),
        word(param::MAX_TEXT_LENGTH),
        word(param::INPUT_TEXT_BUFFER),
        word(param::WORK),
        word(param::HANDLER),
    );
    let (kind, input_method, arg, filter, initial) = (
        word(param::TYPE),
        word(param::INPUT_METHOD),
        word(param::ARG),
        word(param::FILTER),
        word(param::INITIAL_TEXT),
    );
    if kind > TYPE_MAX {
        return ERROR_INVALID_TYPE;
    }
    if languages & !LANGUAGE_MASK != 0 {
        return ERROR_INVALID_SUPPORTED_LANGUAGES;
    }
    if enter_label > ENTER_LABEL_MAX {
        return ERROR_INVALID_ENTER_LABEL;
    }
    if input_method != 0 {
        return ERROR_INVALID_INPUT_METHOD;
    }
    if option & !OPTION_MASK != 0 {
        return ERROR_INVALID_OPTION;
    }
    if max_len == 0 || max_len > MAX_TEXT_LENGTH {
        return ERROR_INVALID_MAX_TEXT_LENGTH;
    }
    if buf == 0 {
        return ERROR_INVALID_INPUT_TEXT_BUFFER;
    }
    if work == 0 || work % 4 != 0 {
        return ERROR_INVALID_WORK;
    }
    if handler == 0 {
        return ERROR_INVALID_HANDLER;
    }
    if ctx
        .read_bytes(a + param::RESERVED, 7)
        .iter()
        .any(|&b| b != 0)
    {
        return ERROR_INVALID_RESERVED;
    }
    if work < buf.saturating_add(2 * (max_len + 1)) && buf < work.saturating_add(WORK_SIZE) {
        return ERROR_INVALID_PARAM;
    }
    let multiline = option & OPTION_MULTILINE != 0;
    let text = if initial == 0 {
        Vec::new()
    } else {
        read_text(ctx, initial, max_len as usize + 1)
    };
    if text.len() > max_len as usize || (!multiline && text.contains(&u16::from(b'\n'))) {
        return ERROR_INVALID_TEXT;
    }
    ctx.write_bytes(work, &vec![0u8; WORK_SIZE as usize]);
    write_text(ctx, buf, &text);
    // Typing done while no keyboard was open was not typed into this one.
    let _ = st.world.poll_text();
    st.ime.session = Some(Session {
        handler,
        arg,
        filter,
        work,
        input_buf: buf,
        max_len: max_len as usize,
        multiline,
        caret: text.len(),
        text,
        open_pending: true,
        preedit: (0, 0, 0),
    });
    st.ime.closed_during = None;
    0
}

/// An `UPDATE_TEXT` for an edit at `index` that changed the length by `change`, leaving the caret
/// at `caret`; `str` points at the game's `inputTextBuffer`, which is written first.
fn text_event(buf: u32, text: &[u16], index: usize, change: i32, caret: usize) -> Event {
    let mut data = [0u32; 10];
    data[0] = caret as u32; // preeditIndex: no pre-edit, so at the caret
    data[1] = 0; // preeditLength
    data[2] = caret as u32;
    data[3] = buf;
    data[4] = index as u32;
    data[5] = change as u32;
    Event {
        id: EVENT_UPDATE_TEXT,
        data,
        text: Some(text.to_vec()),
    }
}

/// SceInt32 sceImeUpdate(void)
pub(super) fn update(ctx: &mut GuestCtx, st: &mut VitaState) {
    let r = update_in(ctx, st);
    ctx.ret(r as u32);
}

fn update_in(ctx: &mut GuestCtx, st: &mut VitaState) -> i32 {
    if st.ime.outbox.is_none() {
        let Some(s) = st.ime.session.as_ref() else {
            return ERROR_NOT_OPENED;
        };
        let (filter, work, buf, max_len, multiline) =
            (s.filter, s.work, s.input_buf, s.max_len, s.multiline);
        let (mut text, mut caret) = (s.text.clone(), s.caret);
        let open_pending = s.open_pending;
        if st.ime.keys.is_none() {
            let keys = st.world.poll_text();
            st.ime.keys = Some(keys);
        }
        let keys = st.ime.keys.clone().unwrap_or_default();
        let mut events = Vec::new();
        if open_pending {
            let mut data = [0u32; 10];
            (data[0], data[1], data[2], data[3]) = PANEL;
            events.push(Event {
                id: EVENT_OPEN,
                data,
                text: None,
            });
        }
        for key in keys {
            let insert: Vec<u16> = match key {
                TextKey::Enter if multiline => vec![u16::from(b'\n')],
                TextKey::Char(c) if c == '\n' && !multiline => Vec::new(),
                TextKey::Char(c) => {
                    let mut units = [0u16; 2];
                    c.encode_utf16(&mut units).to_vec()
                }
                TextKey::Enter => {
                    events.push(Event {
                        id: EVENT_PRESS_ENTER,
                        data: [0; 10],
                        text: None,
                    });
                    continue;
                }
                TextKey::Escape => {
                    events.push(Event {
                        id: EVENT_PRESS_CLOSE,
                        data: [0; 10],
                        text: None,
                    });
                    continue;
                }
                TextKey::Backspace => {
                    if caret > 0 {
                        caret -= 1;
                        text.remove(caret);
                        events.push(text_event(buf, &text, caret, -1, caret));
                    }
                    continue;
                }
                TextKey::Left | TextKey::Right => {
                    let to = if key == TextKey::Left {
                        caret.saturating_sub(1)
                    } else {
                        (caret + 1).min(text.len())
                    };
                    if to != caret {
                        caret = to;
                        let mut data = [0u32; 10];
                        data[0] = caret as u32;
                        events.push(Event {
                            id: EVENT_UPDATE_CARET,
                            data,
                            text: None,
                        });
                    }
                    continue;
                }
            };
            if insert.is_empty() {
                continue;
            }
            // The game's filter sees each typed unit and answers what is inserted instead.
            let insert = if filter == 0 {
                insert
            } else {
                write_text(ctx, work + WORK_FILTER_SRC, &insert);
                let call = GuestCall {
                    entry: filter,
                    args: vec![
                        work + WORK_FILTER_OUT,
                        work + WORK_FILTER_OUT_LEN,
                        work + WORK_FILTER_SRC,
                        insert.len() as u32,
                    ],
                };
                // The answer is read from the bytes captured as the call returned: every filter
                // call of this update reuses the same buffer, so by a replay it holds a later one.
                let capture = (
                    work + WORK_FILTER_OUT_LEN,
                    WORK_FILTER_OUT - WORK_FILTER_OUT_LEN + FILTER_OUT_MAX as u32 * 2,
                );
                let Some(r) = st.call_guest(call, Some(capture)) else {
                    return 0;
                };
                let word = |off: usize| {
                    r.bytes
                        .get(off..off + 2)
                        .map_or(0, |b| u16::from_le_bytes([b[0], b[1]]))
                };
                let n = (r
                    .bytes
                    .get(..4)
                    .map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    as usize)
                    .min(FILTER_OUT_MAX);
                let base = (WORK_FILTER_OUT - WORK_FILTER_OUT_LEN) as usize;
                (0..n).map(|i| word(base + 2 * i)).collect()
            };
            // A full buffer takes no more; the key is dropped without an event.
            if insert.is_empty() || text.len() + insert.len() > max_len {
                continue;
            }
            let at = caret;
            text.splice(at..at, insert.iter().copied());
            caret += insert.len();
            events.push(text_event(buf, &text, at, insert.len() as i32, caret));
        }
        if let Some(s) = st.ime.session.as_mut() {
            s.text = text;
            s.caret = caret;
            s.open_pending = false;
        }
        st.ime.keys = None;
        st.ime.closed_during = None;
        st.ime.outbox = Some(events);
    }
    let events = st.ime.outbox.clone().unwrap_or_default();
    let Some(s) = st
        .ime
        .session
        .as_ref()
        .map(|s| (s.handler, s.arg, s.work))
        .or_else(|| st.ime.last_target())
    else {
        st.ime.outbox = None;
        return 0;
    };
    let (handler, arg, work) = s;
    for (i, e) in events.iter().enumerate() {
        if st.ime.closed_during.is_some_and(|c| c < i) {
            break;
        }
        if let Some(text) = &e.text
            && let Some(buf) = e.data.get(3).copied()
        {
            write_text(ctx, buf, text);
        }
        let mut record = Vec::with_capacity(0x2C);
        record.extend_from_slice(&e.id.to_le_bytes());
        for w in e.data {
            record.extend_from_slice(&w.to_le_bytes());
        }
        ctx.write_bytes(work + WORK_EVENT, &record);
        st.ime.delivering = i;
        if st
            .call_guest(
                GuestCall {
                    entry: handler,
                    args: vec![arg, work + WORK_EVENT],
                },
                None,
            )
            .is_none()
        {
            return 0;
        }
    }
    st.ime.outbox = None;
    st.ime.closed_target = None;
    0
}

impl ImeState {
    /// The handler a delivery in progress is going to, after the handler itself closed the IME:
    /// the remaining replays of that delivery still answer the calls already made.
    fn last_target(&self) -> Option<(u32, u32, u32)> {
        self.closed_target
    }
}

/// SceInt32 sceImeSetText(const SceWChar16 *text, SceUInt32 length)
///
/// Replace the whole text (up to `length` units, or the first NUL) and put the caret at its end.
/// The game's own change is not echoed back as an event.
#[hostcall]
pub(super) fn set_text(ctx: &mut GuestCtx, st: &mut VitaState, text: Ptr, length: u32) -> i32 {
    match st.ime.session.as_mut() {
        None => ERROR_NOT_OPENED,
        Some(_) if text.is_null() => ERROR_INVALID_POINTER,
        Some(s) if length as usize > s.max_len => ERROR_INVALID_TEXT,
        Some(s) => {
            let t = read_text(ctx, text.addr(), length as usize);
            s.caret = t.len();
            write_text(ctx, s.input_buf, &t);
            s.text = t;
            0
        }
    }
}

/// SceInt32 sceImeSetCaret(const SceImeCaret *caret)
///
/// `SceImeCaret` is `{ x, y, height, index }`; the index moves the caret, the rest places the
/// system's caret, which this front end does not draw.
#[hostcall]
pub(super) fn set_caret(ctx: &mut GuestCtx, st: &mut VitaState, caret: Ptr) -> i32 {
    match st.ime.session.as_mut() {
        None => ERROR_NOT_OPENED,
        Some(_) if caret.is_null() => ERROR_INVALID_POINTER,
        Some(s) => {
            let index = ctx.read_u32(caret.addr() + 0xC) as usize;
            if index > s.text.len() {
                ERROR_INVALID_PARAM
            } else {
                s.caret = index;
                0
            }
        }
    }
}

/// SceInt32 sceImeSetPreeditGeometry(const SceImePreeditGeometry *preedit)
///
/// Where the game draws its pre-edit text `{ x, y, height }`, for placing a candidate window.
#[hostcall]
pub(super) fn set_preedit_geometry(ctx: &mut GuestCtx, st: &mut VitaState, g: Ptr) -> i32 {
    match st.ime.session.as_mut() {
        None => ERROR_NOT_OPENED,
        Some(_) if g.is_null() => ERROR_INVALID_POINTER,
        Some(s) => {
            s.preedit = (
                ctx.read_u32(g.addr()),
                ctx.read_u32(g.addr() + 4),
                ctx.read_u32(g.addr() + 8),
            );
            0
        }
    }
}

/// SceInt32 sceImeClose(void)
///
/// Callable from inside the handler: no event after the one being delivered goes out.
#[hostcall]
pub(super) fn close(st: &mut VitaState) -> i32 {
    match st.ime.session.take() {
        None => ERROR_NOT_OPENED,
        Some(s) => {
            if st.ime.outbox.is_some() {
                st.ime.closed_during = Some(st.ime.delivering);
                st.ime.closed_target = Some((s.handler, s.arg, s.work));
            } else {
                st.ime.closed_target = None;
            }
            st.ime.keys = None;
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::{SliceMemory, VFP_ARG_COUNT};
    use crate::world::{CtrlFrame, World};
    use vitaslop_transpiler::abi::REG_COUNT;

    /// A world whose keyboard types `keys` once, AFTER the keyboard opened: the first poll is
    /// the open's own drain of anything typed before it (see `open`), which is empty here.
    struct Keys(Vec<TextKey>, bool);
    impl World for Keys {
        fn monotonic_us(&mut self) -> u64 {
            0
        }
        fn wall_us(&mut self) -> u64 {
            0
        }
        fn poll_ctrl(&mut self, _port: u32) -> CtrlFrame {
            CtrlFrame::default()
        }
        fn fill_random(&mut self, _buf: &mut [u8]) {}
        fn poll_text(&mut self) -> Vec<TextKey> {
            if !std::mem::replace(&mut self.1, true) {
                return Vec::new();
            }
            std::mem::take(&mut self.0)
        }
    }

    const MEM: usize = 0x8000;
    const PARAM: u32 = 0x100;
    const INITIAL: u32 = 0x200;
    const BUF: u32 = 0x300;
    const WORK: u32 = 0x1000;

    struct Run {
        regs: [u32; REG_COUNT],
        vfp: [u32; VFP_ARG_COUNT],
        bytes: Vec<u8>,
        st: VitaState,
    }

    impl Run {
        fn new(keys: Vec<TextKey>) -> Self {
            Run {
                regs: [0; REG_COUNT],
                vfp: [0; VFP_ARG_COUNT],
                bytes: vec![0u8; MEM],
                st: VitaState::new(0, MEM as u32, Box::new(Keys(keys, false))),
            }
        }
        fn call(&mut self, f: fn(&mut GuestCtx, &mut VitaState), args: &[u32]) -> i32 {
            self.regs = [0; REG_COUNT];
            self.regs[..args.len()].copy_from_slice(args);
            {
                let mut mem = SliceMemory(&mut self.bytes);
                let mut ctx = GuestCtx::new(&mut self.regs, &mut self.vfp, &mut mem, 0);
                f(&mut ctx, &mut self.st);
            }
            self.regs[0] as i32
        }
        fn put(&mut self, at: u32, v: u32) {
            self.bytes[at as usize..at as usize + 4].copy_from_slice(&v.to_le_bytes());
        }
        /// A valid parameter block: max 8 units, initial text "ab", a handler.
        fn param(&mut self, option: u32) {
            self.bytes[PARAM as usize..PARAM as usize + 0x40].fill(0);
            self.put(PARAM + param::OPTION, option);
            self.put(PARAM + param::WORK, WORK);
            self.put(PARAM + param::ARG, 0x77);
            self.put(PARAM + param::HANDLER, 0x9001);
            self.put(PARAM + param::INITIAL_TEXT, INITIAL);
            self.put(PARAM + param::MAX_TEXT_LENGTH, 8);
            self.put(PARAM + param::INPUT_TEXT_BUFFER, BUF);
            self.bytes[INITIAL as usize..INITIAL as usize + 6]
                .copy_from_slice(&[b'a', 0, b'b', 0, 0, 0]);
        }
        fn buf_text(&self) -> String {
            let mut out = Vec::new();
            for i in 0.. {
                let a = BUF as usize + 2 * i;
                let u = u16::from_le_bytes([self.bytes[a], self.bytes[a + 1]]);
                if u == 0 {
                    break;
                }
                out.push(u);
            }
            String::from_utf16(&out).unwrap()
        }
    }

    /// Each field is refused with its own error, in order, and a valid block opens once: the
    /// initial text is in the game's buffer and the caret is after it.
    #[test]
    fn open_checks_its_parameters() {
        let mut r = Run::new(vec![]);
        assert_eq!(r.call(open, &[0]), ERROR_INVALID_POINTER);
        r.param(0);
        r.put(PARAM + param::TYPE, 6);
        assert_eq!(r.call(open, &[PARAM]), ERROR_INVALID_TYPE);
        r.param(0x8);
        assert_eq!(r.call(open, &[PARAM]), ERROR_INVALID_OPTION);
        r.param(0);
        r.put(PARAM + param::MAX_TEXT_LENGTH, 0);
        assert_eq!(r.call(open, &[PARAM]), ERROR_INVALID_MAX_TEXT_LENGTH);
        r.param(0);
        r.put(PARAM + param::HANDLER, 0);
        assert_eq!(r.call(open, &[PARAM]), ERROR_INVALID_HANDLER);
        r.param(0);
        r.put(PARAM + param::WORK, WORK + 2);
        assert_eq!(r.call(open, &[PARAM]), ERROR_INVALID_WORK);
        r.param(0);
        r.put(PARAM + param::MAX_TEXT_LENGTH, 1);
        assert_eq!(r.call(open, &[PARAM]), ERROR_INVALID_TEXT);
        r.param(0);
        assert_eq!(r.call(open, &[PARAM]), 0);
        assert_eq!(r.buf_text(), "ab");
        assert_eq!(r.st.ime.session.as_ref().unwrap().caret, 2);
        assert_eq!(r.call(open, &[PARAM]), ERROR_ALREADY_OPENED);
    }

    /// The first update delivers OPEN, then one UPDATE_TEXT per key (caret after the edit, the
    /// edit's index and signed length change), a caret move, and Enter/Escape as their presses;
    /// the delivery stops at the first handler call, which the engine has yet to run.
    #[test]
    fn keys_become_events_on_the_next_update() {
        use TextKey::*;
        let mut r = Run::new(vec![Char('c'), Backspace, Left, Char('X'), Enter, Escape]);
        r.param(0);
        assert_eq!(r.call(open, &[PARAM]), 0);
        assert_eq!(r.call(update, &[]), 0);
        let out =
            r.st.ime
                .outbox
                .clone()
                .expect("the delivery is in progress");
        let ids: Vec<u32> = out.iter().map(|e| e.id).collect();
        assert_eq!(
            ids,
            vec![
                EVENT_OPEN,
                EVENT_UPDATE_TEXT,
                EVENT_UPDATE_TEXT,
                EVENT_UPDATE_CARET,
                EVENT_UPDATE_TEXT,
                EVENT_PRESS_ENTER,
                EVENT_PRESS_CLOSE
            ]
        );
        assert_eq!(
            &out[0].data[..4],
            &[0, 544 - PANEL_HEIGHT, 960, PANEL_HEIGHT]
        );
        let text = |e: &Event| String::from_utf16(e.text.as_ref().unwrap()).unwrap();
        // "ab" + c -> "abc", caret 3, edit at 2, +1
        assert_eq!(
            (
                text(&out[1]),
                out[1].data[2],
                out[1].data[4],
                out[1].data[5] as i32
            ),
            ("abc".into(), 3, 2, 1)
        );
        assert_eq!(out[1].data[3], BUF);
        // backspace -> "ab", caret 2, edit at 2, -1
        assert_eq!(
            (
                text(&out[2]),
                out[2].data[2],
                out[2].data[4],
                out[2].data[5] as i32
            ),
            ("ab".into(), 2, 2, -1)
        );
        // left -> caret 1
        assert_eq!(out[3].data[0], 1);
        // X at 1 -> "aXb", caret 2
        assert_eq!(
            (text(&out[4]), out[4].data[2], out[4].data[4]),
            ("aXb".into(), 2, 1)
        );
        // The session holds the result; the OPEN event's record is in the work buffer.
        assert_eq!(
            r.st.ime.session.as_ref().unwrap().text,
            "aXb".encode_utf16().collect::<Vec<_>>()
        );
        assert_eq!(
            &r.bytes[WORK as usize..WORK as usize + 4],
            &EVENT_OPEN.to_le_bytes()
        );
    }

    /// Enter inserts a newline under MULTILINE; a full buffer takes no more keys.
    #[test]
    fn multiline_enter_and_a_full_buffer() {
        use TextKey::*;
        let mut r = Run::new(vec![
            Enter,
            Char('1'),
            Char('2'),
            Char('3'),
            Char('4'),
            Char('5'),
            Char('6'),
            Char('7'),
        ]);
        r.param(OPTION_MULTILINE);
        assert_eq!(r.call(open, &[PARAM]), 0);
        r.call(update, &[]);
        let s = r.st.ime.session.as_ref().unwrap();
        assert_eq!(String::from_utf16(&s.text).unwrap(), "ab\n12345");
        assert_eq!(s.text.len(), 8);
    }

    /// SetText replaces the text and writes the buffer; SetCaret refuses an index past the end;
    /// Close ends the session and every call after says so.
    #[test]
    fn set_text_set_caret_and_close() {
        let mut r = Run::new(vec![]);
        assert_eq!(r.call(update, &[]), ERROR_NOT_OPENED);
        r.param(0);
        r.call(open, &[PARAM]);
        r.bytes[0x400..0x408].copy_from_slice(&[b'h', 0, b'i', 0, 0, 0, 0, 0]);
        assert_eq!(r.call(set_text, &[0x400, 4]), 0);
        assert_eq!(r.buf_text(), "hi");
        assert_eq!(r.call(set_text, &[0x400, 9]), ERROR_INVALID_TEXT);
        r.put(0x500 + 0xC, 3);
        assert_eq!(r.call(set_caret, &[0x500]), ERROR_INVALID_PARAM);
        r.put(0x500 + 0xC, 1);
        assert_eq!(r.call(set_caret, &[0x500]), 0);
        assert_eq!(r.call(close, &[]), 0);
        assert_eq!(r.call(close, &[]), ERROR_NOT_OPENED);
        assert_eq!(r.call(set_caret, &[0x500]), ERROR_NOT_OPENED);
    }
}
