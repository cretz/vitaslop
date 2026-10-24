//! SceHttp: the HTTP client surface, modelled OFFLINE.
//!
//! # The same boundary SceNet draws, one layer up
//! [`crate::vita::net`] models a console whose network interface is DOWN, and this
//! module has to tell the same story: SceHttp is built on SceNet, so a title that asks
//! the link state, then opens a socket, then issues a request must get one consistent
//! answer from all three. Nothing here opens a host connection, and nothing fabricates
//! a server reply - a title handed an invented response body would act on data nobody
//! sent, and the run would stop being reproducible from its own inputs.
//!
//! # What is real
//! The whole LOCAL object graph, because all of it is the guest's own state:
//! - `sceHttpCreateTemplate` really allocates a template, `CreateConnectionWithURL` a
//!   connection under a valid template, `CreateRequestWithURL` a request under a valid
//!   connection. A parent id that is wrong or already deleted is REFUSED
//!   (`SCE_HTTP_ERROR_INVALID_ID`), so a title that mismanages its handles sees it.
//! - The delete calls really free, and the id ranges are disjoint per kind, so passing a
//!   connection id to `sceHttpDeleteRequest` fails instead of tearing down a stranger.
//! - Timeouts and request headers are really recorded against the object the guest set
//!   them on. Nothing reads them back here today, but they are what the guest said, and
//!   the alternative is dropping them and calling it success.
//!
//! # What reports the link being down
//! Exactly one call: `sceHttpSendRequest`, with `SCE_HTTP_ERROR_RESOLVER_ENODNS` - "no
//! DNS server", which is precisely a console with no interface up, and the same failure
//! `sceNetResolverStartNtoa` gives one layer down. Creating the objects is local work
//! that genuinely succeeds on such a console; the send is the first call that needs a
//! peer, so it is the first that can honestly fail.
//!
//! Everything that reads a RESPONSE - the status code, the content length, the body -
//! then reports `SCE_HTTP_ERROR_BEFORE_SEND`, which is the library's own error for
//! being asked about a reply to a request that has not been sent. That is true here:
//! the send failed, so no request was ever sent.

use crate::host::{GuestCtx, HttpKind, Ptr, VitaState};
use crate::hostcall;

/// `SceHttpErrorCode` values this surface uses, from `psp2/net/http.h`.
const SCE_HTTP_ERROR_INVALID_ID: i32 = 0x8043_1100u32 as i32;
const SCE_HTTP_ERROR_INVALID_VALUE: i32 = 0x8043_11FEu32 as i32;
const SCE_HTTP_ERROR_BEFORE_SEND: i32 = 0x8043_1065u32 as i32;
const SCE_HTTP_ERROR_OUT_OF_MEMORY: i32 = 0x8043_1022u32 as i32;
const SCE_HTTP_ERROR_INVALID_URL: i32 = 0x8043_3060u32 as i32;
const SCE_HTTP_ERROR_UNKNOWN_SCHEME: i32 = 0x8043_1061u32 as i32;
const SCE_HTTP_ERROR_UNKNOWN_METHOD: i32 = 0x8043_106Bu32 as i32;
const SCE_HTTP_ERROR_PARSE_HTTP_NOT_FOUND: i32 = 0x8043_2025u32 as i32;
const SCE_HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE: i32 = 0x8043_2060u32 as i32;
const SCE_HTTP_ERROR_PARSE_HTTP_INVALID_VALUE: i32 = 0x8043_21FEu32 as i32;
/// The last `SceHttpMethods` value (CONNECT).
const METHOD_MAX: i32 = 7;
/// `SCE_HTTP_ERROR_RESOLVER_ENODNS`: no DNS server is reachable. The link is down, so
/// there is none - this is the failure a real console produces, not a stand-in.
const SCE_HTTP_ERROR_RESOLVER_ENODNS: i32 = 0x8043_6002u32 as i32;

/// How much of a URL or user-agent string is worth keeping. These are diagnostics (they
/// name the object in a trace); a title's URLs are far shorter than this.
const LABEL_MAX: usize = 256;

/// int sceHttpCreateTemplate(const char *userAgent, int httpVer, int autoProxyConf)
///
/// Allocates a template - the object every connection inherits its settings from. This
/// is pure local bookkeeping on hardware too (no socket is opened until a request is
/// sent), so it succeeds on a console with no link.
#[hostcall]
pub(super) fn create_template(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    user_agent: Ptr,
    _http_ver: i32,
    _auto_proxy: i32,
) -> i32 {
    let ua = if user_agent.is_null() {
        String::new()
    } else {
        ctx.read_cstr(user_agent.addr(), LABEL_MAX)
    };
    st.http_create(HttpKind::Template, 0, &ua, 0)
}

/// int sceHttpDeleteTemplate(int tmplId)
#[hostcall]
pub(super) fn delete_template(st: &mut VitaState, tmpl: i32) -> i32 {
    delete_of_kind(st, tmpl, HttpKind::Template)
}

/// int sceHttpCreateConnectionWithURL(int tmplId, const char *url, int enableKeepalive)
///
/// A connection OBJECT, not a connection: SceHttp resolves and connects lazily, at send
/// time. So this succeeds with the link down, and the URL is recorded against it.
#[hostcall]
pub(super) fn create_connection_with_url(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    tmpl: i32,
    url: Ptr,
    _keepalive: i32,
) -> i32 {
    if !st.http_is(tmpl, HttpKind::Template) {
        SCE_HTTP_ERROR_INVALID_ID
    } else if url.is_null() {
        SCE_HTTP_ERROR_INVALID_VALUE
    } else {
        let url = ctx.read_cstr(url.addr(), LABEL_MAX);
        st.http_create(HttpKind::Connection, tmpl, &url, 0)
    }
}

/// int sceHttpCreateConnection(int tmplId, const char *serverName, const char *scheme,
///                             unsigned short port, int enableKeepalive)
///
/// The same lazy connection object as [`create_connection_with_url`], named by its parts
/// instead of a URL. It is labelled with the URL those parts spell, so a trace reads the same
/// whichever form the title used. The scheme must be `http` or `https` in any case; the port is
/// kept as given, 0 included (it means the scheme's default when a request is sent).
#[hostcall]
pub(super) fn create_connection(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    tmpl: i32,
    server: Ptr,
    scheme: Ptr,
    port: u32,
    _keepalive: i32,
) -> i32 {
    if !st.http_is(tmpl, HttpKind::Template) {
        SCE_HTTP_ERROR_INVALID_ID
    } else if server.is_null() || scheme.is_null() {
        SCE_HTTP_ERROR_INVALID_VALUE
    } else {
        let server = ctx.read_cstr(server.addr(), LABEL_MAX);
        let scheme = ctx.read_cstr(scheme.addr(), 0x20);
        if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
            SCE_HTTP_ERROR_UNKNOWN_SCHEME
        } else {
            let url = format!("{scheme}://{server}:{}", port & 0xFFFF);
            st.http_create(HttpKind::Connection, tmpl, &url, 0)
        }
    }
}

/// int sceHttpDeleteConnection(int connId)
#[hostcall]
pub(super) fn delete_connection(st: &mut VitaState, conn: i32) -> i32 {
    delete_of_kind(st, conn, HttpKind::Connection)
}

/// int sceHttpCreateRequestWithURL(int connId, int method, const char *url,
///                                 unsigned long long contentLength)
///
/// `contentLength` is a 64-bit trailing argument and nothing offline consumes it, so it
/// is not marshalled - it is the LAST argument, so skipping it shifts nothing else.
#[hostcall]
pub(super) fn create_request_with_url(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    conn: i32,
    method: i32,
    url: Ptr,
) -> i32 {
    if !st.http_is(conn, HttpKind::Connection) {
        SCE_HTTP_ERROR_INVALID_ID
    } else if url.is_null() {
        SCE_HTTP_ERROR_INVALID_VALUE
    } else {
        let url = ctx.read_cstr(url.addr(), LABEL_MAX);
        st.http_create(HttpKind::Request, conn, &url, method)
    }
}

/// int sceHttpCreateRequest(int connId, int method, const char *path,
///                          unsigned long long contentLength)
///
/// [`create_request_with_url`] with a path on the connection's server instead of a whole URL;
/// the trailing 64-bit length is skipped for the same reason. A method outside `SceHttpMethods`
/// is refused before the path is looked at.
#[hostcall]
pub(super) fn create_request(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    conn: i32,
    method: i32,
    path: Ptr,
) -> i32 {
    if !st.http_is(conn, HttpKind::Connection) {
        SCE_HTTP_ERROR_INVALID_ID
    } else if !(0..=METHOD_MAX).contains(&method) {
        SCE_HTTP_ERROR_UNKNOWN_METHOD
    } else if path.is_null() {
        SCE_HTTP_ERROR_INVALID_VALUE
    } else {
        let path = ctx.read_cstr(path.addr(), LABEL_MAX);
        st.http_create(HttpKind::Request, conn, &path, method)
    }
}

/// int sceHttpDeleteRequest(int reqId)
#[hostcall]
pub(super) fn delete_request(st: &mut VitaState, req: i32) -> i32 {
    delete_of_kind(st, req, HttpKind::Request)
}

/// Free `id` only if it is of `kind`. The id ranges are disjoint per kind, so this
/// catches a template id passed to `sceHttpDeleteRequest` rather than freeing whatever
/// happens to share the number.
fn delete_of_kind(st: &mut VitaState, id: i32, kind: HttpKind) -> i32 {
    if st.http_is(id, kind) && st.http_delete(id) {
        0
    } else {
        SCE_HTTP_ERROR_INVALID_ID
    }
}

/// int sceHttpSetConnectTimeOut(int id, unsigned int usec)
/// int sceHttpSetSendTimeOut(int id, unsigned int usec)
/// int sceHttpSetRecvTimeOut(int id, unsigned int usec)
///
/// Each records the value against the object it was set on (any of the three kinds -
/// the setting is inherited down the graph on hardware). `which` is the timeout slot,
/// supplied by the dispatch arm rather than by the guest, so the three NIDs share one
/// body without any of them guessing which one it is.
pub(super) fn set_timeout(ctx: &mut GuestCtx, st: &mut VitaState, which: usize) {
    let (id, usec) = (ctx.arg(0) as i32, ctx.arg(1));
    let r = if st.http_set_timeout(id, which, usec) {
        0
    } else {
        SCE_HTTP_ERROR_INVALID_ID
    };
    ctx.ret(r as u32);
}

/// int sceHttpUriEscape(char *out, unsigned int *require, unsigned int prepare, const char *in)
///
/// Percent-encodes every byte but RFC 3986's unreserved set (`A-Z a-z 0-9 - . _ ~`), with
/// UPPERCASE hex; `*require` is the escaped length INCLUDING the NUL. With no `out` it is a size
/// query. A `prepare` too small for the result leaves `out` untouched; the error answered then
/// (OUT_OF_SIZE) is the library's size error by name - the code for that case is not published.
#[hostcall]
pub(super) fn uri_escape(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    out: Ptr,
    require: Ptr,
    prepare: u32,
    input: Ptr,
) -> i32 {
    const SCE_HTTP_ERROR_OUT_OF_SIZE: i32 = 0x8043_1104u32 as i32;
    if input.is_null() {
        SCE_HTTP_ERROR_INVALID_VALUE
    } else {
        // Raw bytes, not a lossy UTF-8 string: a byte outside ASCII is exactly what gets escaped.
        let raw = ctx.read_bytes(input.addr(), 8192);
        let src = &raw[..raw.iter().position(|&b| b == 0).unwrap_or(raw.len())];
        let mut esc = String::with_capacity(src.len() * 3 + 1);
        for &b in src {
            if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
                esc.push(b as char);
            } else {
                esc.push_str(&format!("%{b:02X}"));
            }
        }
        esc.push('\0');
        if !require.is_null() {
            ctx.write_u32(require.addr(), esc.len() as u32);
        }
        if out.is_null() {
            0
        } else if (prepare as usize) < esc.len() {
            SCE_HTTP_ERROR_OUT_OF_SIZE
        } else {
            ctx.write_bytes(out.addr(), esc.as_bytes());
            0
        }
    }
}

/// int sceHttpSetRequestContentLength(int reqId, unsigned long long contentLength)
///
/// A request's body length, which only a send consumes - and offline no send reaches the
/// wire - so it is validated against the id and not otherwise kept.
#[hostcall]
pub(super) fn set_request_content_length(st: &mut VitaState, req: i32) -> i32 {
    if st.http_is(req, HttpKind::Request) {
        0
    } else {
        SCE_HTTP_ERROR_INVALID_ID
    }
}

/// int sceHttpAddRequestHeader(int id, const char *name, const char *value,
///                             unsigned int mode)
#[hostcall]
pub(super) fn add_request_header(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    id: i32,
    name: Ptr,
    value: Ptr,
    _mode: u32,
) -> i32 {
    if name.is_null() {
        SCE_HTTP_ERROR_INVALID_VALUE
    } else {
        let name = ctx.read_cstr(name.addr(), LABEL_MAX);
        let value = if value.is_null() {
            String::new()
        } else {
            ctx.read_cstr(value.addr(), LABEL_MAX)
        };
        if st.http_add_header(id, &name, &value) {
            0
        } else {
            SCE_HTTP_ERROR_INVALID_ID
        }
    }
}

/// int sceHttpSendRequest(int reqId, const void *postData, unsigned int size)
///
/// THE call that needs the network, and the only one here that reports it missing. See
/// the module header for why the failure is the resolver's and not, say, a timeout: a
/// console with the interface down has no DNS server configured, so the lookup that
/// precedes the connect is what fails, immediately rather than after 120 seconds.
#[hostcall]
pub(super) fn send_request(st: &mut VitaState, req: i32) -> i32 {
    if st.http_is(req, HttpKind::Request) {
        SCE_HTTP_ERROR_RESOLVER_ENODNS
    } else {
        SCE_HTTP_ERROR_INVALID_ID
    }
}

/// int sceHttpAbortRequest(int reqId)
///
/// Aborting a request that is not in flight genuinely succeeds: there is nothing to
/// interrupt, and the call's contract is "this request will not complete", which is
/// already true. Titles reach this from the cleanup path a failed send sends them down.
#[hostcall]
pub(super) fn abort_request(st: &mut VitaState, req: i32) -> i32 {
    if st.http_is(req, HttpKind::Request) {
        0
    } else {
        SCE_HTTP_ERROR_INVALID_ID
    }
}

/// int sceHttpGetStatusCode(int reqId, int *statusCode)
///
/// There is no response. The out-param is written anyway, because a caller that ignores
/// the return code would otherwise read its own uninitialised stack as a status - and a
/// stack word that happens to be 200 is the worst possible outcome here.
#[hostcall]
pub(super) fn get_status_code(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    req: i32,
    status: Ptr,
) -> i32 {
    if !st.http_is(req, HttpKind::Request) {
        SCE_HTTP_ERROR_INVALID_ID
    } else {
        if !status.is_null() {
            ctx.write_u32(status.addr(), 0);
        }
        SCE_HTTP_ERROR_BEFORE_SEND
    }
}

/// int sceHttpGetResponseContentLength(int reqId, unsigned long long *contentLength)
#[hostcall]
pub(super) fn get_response_content_length(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    req: i32,
    len: Ptr,
) -> i32 {
    if !st.http_is(req, HttpKind::Request) {
        SCE_HTTP_ERROR_INVALID_ID
    } else {
        if !len.is_null() {
            ctx.write_u32(len.addr(), 0);
            ctx.write_u32(len.addr() + 4, 0);
        }
        SCE_HTTP_ERROR_BEFORE_SEND
    }
}

/// int sceHttpReadData(int reqId, void *data, unsigned int size)
///
/// No body exists to read. This returns the error rather than 0 ("end of body"), which
/// matters: a title that saw 0 would conclude it had received a complete, EMPTY
/// document and act on it, where the error sends it down its no-network path.
#[hostcall]
pub(super) fn read_data(st: &mut VitaState, req: i32) -> i32 {
    if st.http_is(req, HttpKind::Request) {
        SCE_HTTP_ERROR_BEFORE_SEND
    } else {
        SCE_HTTP_ERROR_INVALID_ID
    }
}

/// int sceHttpGetAllResponseHeaders(int reqId, char **header, unsigned int *headerSize)
///
/// The raw header block of the response. There is none - the send failed, so the request was
/// never sent - and the library says so with BEFORE_SEND, leaving both out-params as they were.
#[hostcall]
pub(super) fn get_all_response_headers(
    st: &mut VitaState,
    req: i32,
    header: Ptr,
    size: Ptr,
) -> i32 {
    if header.is_null() || size.is_null() {
        SCE_HTTP_ERROR_INVALID_VALUE
    } else if !st.http_is(req, HttpKind::Request) {
        SCE_HTTP_ERROR_INVALID_ID
    } else {
        SCE_HTTP_ERROR_BEFORE_SEND
    }
}

/// int sceHttpParseResponseHeader(const char *header, unsigned int headerLen,
///                                const char *fieldStr, const char **fieldValue,
///                                unsigned int *valueLen)
///
/// A pure parser over the caller's own bytes - see [`find_header_field`] for the rules. The value
/// comes back as a pointer INTO `header` (not copied, not terminated) and the return value is the
/// offset just past the line that held it, so a caller finds repeats by advancing by it. An
/// empty value is a NULL pointer, a length of 0 and a return of 0.
#[hostcall]
pub(super) fn parse_response_header(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    header: Ptr,
    header_len: u32,
    field: Ptr,
    value_out: Ptr,
    len_out: Ptr,
) -> i32 {
    parse_response_header_in(ctx, header, header_len, field, value_out, len_out)
}

fn parse_response_header_in(
    ctx: &mut GuestCtx,
    header: Ptr,
    header_len: u32,
    field: Ptr,
    value_out: Ptr,
    len_out: Ptr,
) -> i32 {
    if header.is_null() {
        return SCE_HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE;
    }
    if field.is_null() || value_out.is_null() || len_out.is_null() {
        return SCE_HTTP_ERROR_PARSE_HTTP_INVALID_VALUE;
    }
    if header_len == 0 {
        return SCE_HTTP_ERROR_PARSE_HTTP_NOT_FOUND;
    }
    let name = cbytes(ctx, field.addr(), 0xFFF);
    let bytes = ctx.read_bytes(header.addr(), header_len as usize);
    match find_header_field(&bytes, &name) {
        None => SCE_HTTP_ERROR_PARSE_HTTP_NOT_FOUND,
        Some(f) if f.len == 0 => {
            ctx.write_u32(value_out.addr(), 0);
            ctx.write_u32(len_out.addr(), 0);
            0
        }
        Some(f) => {
            ctx.write_u32(value_out.addr(), header.addr() + f.start as u32);
            ctx.write_u32(len_out.addr(), f.len as u32);
            f.line_end as i32
        }
    }
}

/// Where a header field's value sits in a header block: its first byte, its length, and the
/// offset one past the newline that ends it.
#[derive(Debug, PartialEq, Eq)]
struct HeaderField {
    start: usize,
    len: usize,
    line_end: usize,
}

/// ASCII whitespace as the C library classifies it, vertical tab included.
fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r')
}

/// The first field named `name` in `h`:
/// - a name matches only at the start of a line (after an LF, or at 0), case-insensitively, and
///   only when a `:` follows it directly; a line that starts with whitespace never matches;
/// - whitespace after the `:` is skipped, and a newline met while skipping is stepped over and
///   ends the skip, so an empty value is followed by the next line's bytes;
/// - the value runs to the next LF that is not followed by a space or tab (folded lines stay
///   inside it), less one CR directly before that LF.
fn find_header_field(h: &[u8], name: &[u8]) -> Option<HeaderField> {
    let f = name.len();
    for p in 0..h.len() {
        if (p > 0 && h[p - 1] != b'\n') || is_space(h[p]) || f >= h.len() - p {
            continue;
        }
        if !h[p..p + f].eq_ignore_ascii_case(name) || h[p + f] != b':' {
            continue;
        }
        let mut v = p + f + 1;
        while v < h.len() && is_space(h[v]) {
            v += 1;
            if h[v - 1] == b'\n' {
                break;
            }
        }
        let start = v;
        let (mut value_end, mut line_end) = (h.len(), h.len());
        let mut n = start;
        while n < h.len() {
            if h[n] == b'\n' && !(n + 1 < h.len() && matches!(h[n + 1], b' ' | b'\t')) {
                line_end = n + 1;
                value_end = if n > start && h[n - 1] == b'\r' {
                    n - 1
                } else {
                    n
                };
                break;
            }
            n += 1;
        }
        return Some(HeaderField {
            start,
            len: value_end.saturating_sub(start),
            line_end,
        });
    }
    None
}

/// `SceHttpUriElement` (0x2C bytes): `opaque`, seven string pointers, then the port.
#[derive(Default)]
struct UriElement {
    opaque: bool,
    scheme: Vec<u8>,
    username: Vec<u8>,
    password: Vec<u8>,
    hostname: Vec<u8>,
    path: Vec<u8>,
    query: Vec<u8>,
    fragment: Vec<u8>,
    port: u16,
}

/// `SCE_HTTP_URI_BUILD_WITH_*`.
const URI_SCHEME: u32 = 0x01;
const URI_HOSTNAME: u32 = 0x02;
const URI_PORT: u32 = 0x04;
const URI_PATH: u32 = 0x08;
const URI_USERNAME: u32 = 0x10;
const URI_PASSWORD: u32 = 0x20;
const URI_QUERY: u32 = 0x40;
const URI_FRAGMENT: u32 = 0x80;

/// The URI `e` spells under `option`. The authority's `//` follows from `opaque` alone, whatever
/// the bits; a port equal to the scheme's default (443 for https, 80 for http) is left out; the
/// query and fragment carry their own `?` and `#`, as a parse stores them.
fn build_uri(e: &UriElement, option: u32) -> Vec<u8> {
    let mut out = Vec::new();
    if option & URI_SCHEME != 0 && !e.scheme.is_empty() {
        out.extend_from_slice(&e.scheme);
        out.push(b':');
    }
    if !e.opaque {
        out.extend_from_slice(b"//");
    }
    let user = option & URI_USERNAME != 0 && !e.username.is_empty();
    let pass = option & URI_PASSWORD != 0 && !e.password.is_empty();
    if user {
        out.extend_from_slice(&e.username);
    }
    if pass {
        out.push(b':');
        out.extend_from_slice(&e.password);
    }
    if user || pass {
        out.push(b'@');
    }
    if option & URI_HOSTNAME != 0 {
        out.extend_from_slice(&e.hostname);
    }
    if option & URI_PORT != 0 && e.port != 0 {
        let scheme = e.scheme.to_ascii_lowercase();
        let default = match () {
            _ if scheme.len() > 0x20 => 0,
            _ if scheme.starts_with(b"https") => 443,
            _ if scheme.starts_with(b"http") => 80,
            _ => 0,
        };
        if e.port != default {
            out.extend_from_slice(format!(":{}", e.port).as_bytes());
        }
    }
    for (bit, part) in [
        (URI_PATH, &e.path),
        (URI_QUERY, &e.query),
        (URI_FRAGMENT, &e.fragment),
    ] {
        if option & bit != 0 {
            out.extend_from_slice(part);
        }
    }
    out
}

/// The bytes of a C string, up to `max` (empty for a null pointer).
fn cbytes(ctx: &GuestCtx, addr: u32, max: usize) -> Vec<u8> {
    if addr == 0 {
        return Vec::new();
    }
    let mut raw = ctx.read_bytes(addr, max);
    if let Some(n) = raw.iter().position(|&b| b == 0) {
        raw.truncate(n);
    }
    raw
}

/// int sceHttpUriBuild(char *out, unsigned int *require, unsigned int prepare,
///                     const SceHttpUriElement *srcElement, unsigned int option)
///
/// The two-call protocol of [`uri_escape`]: `*require` - written whenever it is given, a
/// too-small buffer included - is the length with its NUL; no `out` is a size query; a `prepare`
/// short of that is the library's OUT_OF_MEMORY and leaves `out` alone.
#[hostcall]
pub(super) fn uri_build(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    out: Ptr,
    require: Ptr,
    prepare: u32,
    src: Ptr,
    option: u32,
) -> i32 {
    uri_build_in(ctx, out, require, prepare, src, option)
}

fn uri_build_in(
    ctx: &mut GuestCtx,
    out: Ptr,
    require: Ptr,
    prepare: u32,
    src: Ptr,
    option: u32,
) -> i32 {
    if src.is_null() {
        return SCE_HTTP_ERROR_INVALID_URL;
    }
    if out.is_null() && require.is_null() {
        return SCE_HTTP_ERROR_INVALID_VALUE;
    }
    let a = src.addr();
    let field = |ctx: &GuestCtx, off: u32| cbytes(ctx, ctx.read_u32(a + off), 0x4000);
    let e = UriElement {
        opaque: ctx.read_u32(a) != 0,
        scheme: field(ctx, 0x04),
        username: field(ctx, 0x08),
        password: field(ctx, 0x0C),
        hostname: field(ctx, 0x10),
        path: field(ctx, 0x14),
        query: field(ctx, 0x18),
        fragment: field(ctx, 0x1C),
        port: ctx.read_u16(a + 0x20),
    };
    let mut uri = build_uri(&e, option);
    uri.push(0);
    if !require.is_null() {
        ctx.write_u32(require.addr(), uri.len() as u32);
    }
    if out.is_null() {
        0
    } else if (prepare as usize) < uri.len() {
        SCE_HTTP_ERROR_OUT_OF_MEMORY
    } else {
        ctx.write_bytes(out.addr(), &uri);
        0
    }
}

/// int sceHttpsLoadCert(int caCertNum, const SceHttpsData **caList,
///                      const SceHttpsData *cert, const SceHttpsData *privKey)
///
/// Loading certificates into the library's own store is local work with no peer
/// involved, and it succeeds - the certs are simply never used, because no handshake
/// ever happens. We do not parse them: nothing offline can consume the result, and a
/// parser that rejected a cert hardware accepts would fail a call that works.
#[hostcall]
pub(super) fn ssl_load_cert(_st: &mut VitaState) -> i32 {
    0
}

/// int sceHttpsSetSslCallback(int id, SceHttpsCallback cbfunc, void *userArg)
///
/// Registers the per-request certificate-verification hook. It is stored on hardware
/// and invoked during the handshake; there is no handshake here, so it is registered
/// against a valid id and never fires. An id that is not a real object is still
/// refused.
#[hostcall]
pub(super) fn ssl_set_ssl_callback(st: &mut VitaState, id: i32) -> i32 {
    if st.http_exists(id) {
        0
    } else {
        SCE_HTTP_ERROR_INVALID_ID
    }
}

/// int sceHttpsGetSslError(int id, int *errNum, unsigned int *detail)
///
/// A title calls this after a failed send to ask "was it the certificate?". Here it was
/// not, and saying so is the accurate answer, not a hollow one: no handshake was
/// attempted, so no SSL error exists. The failure the title should act on is the one
/// `sceHttpSendRequest` already returned. Both out-params are written for the same
/// reason as in [`get_status_code`].
#[hostcall]
pub(super) fn ssl_get_ssl_error(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    id: i32,
    err_num: Ptr,
    detail: Ptr,
) -> i32 {
    if !st.http_exists(id) {
        SCE_HTTP_ERROR_INVALID_ID
    } else {
        if !err_num.is_null() {
            ctx.write_u32(err_num.addr(), 0);
        }
        if !detail.is_null() {
            ctx.write_u32(detail.addr(), 0);
        }
        0
    }
}

// --- The cookie jar ---------------------------------------------------------------
//
// A jar is filled by RESPONSES: the library reads `Set-Cookie` off a reply and stores it
// against the URL's domain. No reply ever arrives here (`sceHttpSendRequest` fails with the
// link down), so the jar is empty for the whole run - not broken, EMPTY, which is the state
// a console is in before its first successful request too. The switch that turns it on and
// the callback that would filter incoming cookies are local settings and are treated as
// such.

/// int sceHttpSetCookieEnabled(int id, int enable)
///
/// Really recorded against the object, and really read back by [`get_cookie_enabled`]. This
/// is the guest's own setting, so dropping it and reporting success would be the one thing
/// this module does not do.
#[hostcall]
pub(super) fn set_cookie_enabled(st: &mut VitaState, id: i32, enable: i32) -> i32 {
    if st.http_set_cookies_enabled(id, enable != 0) {
        0
    } else {
        SCE_HTTP_ERROR_INVALID_ID
    }
}

/// int sceHttpGetCookieEnabled(int id, int *enable)
#[hostcall]
pub(super) fn get_cookie_enabled(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    id: i32,
    enable: Ptr,
) -> i32 {
    match st.http_cookies_enabled(id) {
        None => SCE_HTTP_ERROR_INVALID_ID,
        Some(on) => {
            if !enable.is_null() {
                ctx.write_u32(enable.addr(), on as u32);
            }
            0
        }
    }
}

/// int sceHttpSetCookieRecvCallback(int id, SceHttpCookieRecvCallback cbfunc, void *userArg)
///
/// Registers the hook the library calls for each `Set-Cookie` header in a response, so the
/// title can decide whether to keep it. Registered against a valid id and never invoked,
/// because no response header ever arrives - the same position [`ssl_set_ssl_callback`] is
/// in, and for the same reason.
///
/// The callback is deliberately NOT stored. Nothing can ever call it, and holding it would
/// suggest otherwise; what the call has to get right is refusing an id that names no object.
#[hostcall]
pub(super) fn set_cookie_recv_callback(st: &mut VitaState, id: i32) -> i32 {
    if st.http_exists(id) {
        0
    } else {
        SCE_HTTP_ERROR_INVALID_ID
    }
}

/// int sceHttpSetAuthInfoCallback(int id, SceHttpAuthInfoCallback cbfunc, void *userArg)
/// int sceHttpSetRedirectCallback(int id, SceHttpRedirectCallback cbfunc, void *userArg)
///
/// The hooks the library calls when a response asks for credentials (401/407) or redirects
/// (3xx). Both are invoked only while a response is being processed, and none ever arrives, so
/// like [`set_cookie_recv_callback`] they are accepted against a live id and not stored.
#[hostcall]
pub(super) fn set_response_callback(st: &mut VitaState, id: i32) -> i32 {
    if st.http_exists(id) {
        0
    } else {
        SCE_HTTP_ERROR_INVALID_ID
    }
}

/// int sceHttpGetCookie(const char *url, char *cookie, unsigned int *cookieLength,
///                      unsigned int prepare, int secure)
///
/// Fetch the cookies stored for `url` as one header-shaped string. The jar is empty (see
/// above), so the answer is a length of ZERO and nothing written - which is exactly what the
/// library reports for a URL it holds no cookie for, whether or not it has ever been online.
///
/// `prepare` selects the two-call protocol: non-zero asks only for the LENGTH the caller must
/// allocate, zero asks for the bytes. Both answers are the same here, and both write the
/// length, because a caller that skipped the prepare call still reads `*cookieLength` to know
/// how much it got. Writing it is what stops a title reading an uninitialised stack word as a
/// cookie length and walking a buffer that was never filled.
#[hostcall]
pub(super) fn get_cookie(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    _url: Ptr,
    _cookie: Ptr,
    cookie_length: Ptr,
    _prepare: u32,
    _secure: i32,
) -> i32 {
    if cookie_length.is_null() {
        SCE_HTTP_ERROR_INVALID_VALUE
    } else {
        ctx.write_u32(cookie_length.addr(), 0);
        0
    }
}

#[cfg(test)]
mod tests {
    //! Driven through the real host-call entry points, so the argument marshalling (including
    //! the stack-passed fifth argument of `sceHttpCreateConnection`) is covered with the logic.

    use super::*;
    use crate::host::{SliceMemory, VFP_ARG_COUNT};
    use crate::world::DeterministicWorld;
    use vitaslop_transpiler::abi::REG_COUNT;

    type Entry = fn(&mut GuestCtx, &mut VitaState);

    const STACK: u32 = 0x300;

    struct Run {
        regs: [u32; REG_COUNT],
        vfp: [u32; VFP_ARG_COUNT],
        bytes: Vec<u8>,
        st: VitaState,
    }

    impl Run {
        fn new() -> Self {
            Run {
                regs: [0; REG_COUNT],
                vfp: [0; VFP_ARG_COUNT],
                bytes: vec![0u8; 0x400],
                st: VitaState::new(0, 0x400, Box::new(DeterministicWorld::default())),
            }
        }

        fn cstr(&mut self, addr: u32, s: &str) -> u32 {
            self.bytes[addr as usize..addr as usize + s.len()].copy_from_slice(s.as_bytes());
            self.bytes[addr as usize + s.len()] = 0;
            addr
        }

        /// Call `f` with `args` in r0..r3 and the rest on the stack, as AAPCS passes them.
        fn call(&mut self, f: Entry, args: &[u32]) -> i32 {
            self.regs = [0; REG_COUNT];
            for (i, a) in args.iter().enumerate() {
                if i < 4 {
                    self.regs[i] = *a;
                } else {
                    let at = (STACK + 4 * (i as u32 - 4)) as usize;
                    self.bytes[at..at + 4].copy_from_slice(&a.to_le_bytes());
                }
            }
            self.regs[13] = STACK;
            {
                let mut mem = SliceMemory(&mut self.bytes);
                let mut ctx = GuestCtx::new(&mut self.regs, &mut self.vfp, &mut mem, 0);
                f(&mut ctx, &mut self.st);
            }
            self.regs[0] as i32
        }
    }

    /// A connection built from parts hangs off its template and is labelled with the URL the
    /// parts spell; a request built from a path hangs off that connection. Each refuses a
    /// parent of the wrong level.
    #[test]
    fn a_connection_and_request_built_from_parts_join_the_graph() {
        let mut r = Run::new();
        let ua = r.cstr(0x10, "agent");
        let tmpl = r.call(create_template, &[ua, 2, 0]);
        assert!(tmpl > 0);
        let server = r.cstr(0x40, "example.com");
        let scheme = r.cstr(0x80, "https");
        let conn = r.call(create_connection, &[tmpl as u32, server, scheme, 443, 1]);
        assert!(conn > 0 && conn != tmpl);
        assert!(r.st.http_is(conn, HttpKind::Connection));
        assert!(
            r.st.http_report()
                .iter()
                .any(|l| l.contains("\"https://example.com:443\""))
        );

        let path = r.cstr(0xC0, "/index.html");
        let req = r.call(create_request, &[conn as u32, 0, path]);
        assert!(r.st.http_is(req, HttpKind::Request));

        // A request id is not a template, nor a template id a connection.
        assert_eq!(
            r.call(create_connection, &[req as u32, server, scheme, 443, 1]),
            SCE_HTTP_ERROR_INVALID_ID
        );
        assert_eq!(
            r.call(create_request, &[tmpl as u32, 0, path]),
            SCE_HTTP_ERROR_INVALID_ID
        );
        // Missing names are values, not ids.
        assert_eq!(
            r.call(create_connection, &[tmpl as u32, 0, scheme, 443, 1]),
            SCE_HTTP_ERROR_INVALID_VALUE
        );
        assert_eq!(
            r.call(create_request, &[conn as u32, 0, 0]),
            SCE_HTTP_ERROR_INVALID_VALUE
        );
    }

    /// A field is found by name at a line start, case-insensitively, and its value excludes the
    /// CR of its CRLF; the return is the offset past that line, which is what lets a caller
    /// step over it to the next occurrence.
    #[test]
    fn a_header_field_is_found_at_a_line_start() {
        let h = b"Content-Type: text/html\r\nServer: nginx\r\nX-A: 1\r\nx-a: 2\r\n";
        assert_eq!(
            find_header_field(h, b"content-type"),
            Some(HeaderField {
                start: 14,
                len: 9,
                line_end: 25
            })
        );
        let s = find_header_field(h, b"SERVER").unwrap();
        assert_eq!(&h[s.start..s.start + s.len], b"nginx");
        assert_eq!(s.line_end, 40);
        // Repeats: the first, then from past it, the second.
        let a = find_header_field(h, b"x-a").unwrap();
        assert_eq!(&h[a.start..a.start + a.len], b"1");
        let b = find_header_field(&h[a.line_end..], b"x-a").unwrap();
        assert_eq!(&h[a.line_end + b.start..a.line_end + b.start + b.len], b"2");
        // Mid-line, missing, or a space before the colon: not a match.
        assert_eq!(find_header_field(b"X-Server: a\r\n", b"server"), None);
        assert_eq!(find_header_field(b"Server : a\r\n", b"server"), None);
        assert_eq!(find_header_field(h, b"cookie"), None);
    }

    /// LF-only endings, folded continuation lines kept inside the value, an unterminated last
    /// line, and an empty value (whose skip steps onto the next line).
    #[test]
    fn header_values_fold_and_end_where_the_line_does() {
        let f = find_header_field(b"A: 1\n  2\r\nB: 3\n", b"a").unwrap();
        assert_eq!((f.start, f.len, f.line_end), (3, 5, 10));
        let g = find_header_field(b"A: tail", b"a").unwrap();
        assert_eq!((g.start, g.len, g.line_end), (3, 4, 7));
        let e = find_header_field(b"A:\r\nB: 3\r\n", b"a").unwrap();
        assert_eq!(e.start, 4);
    }

    /// The guest-facing call: the value is a pointer INTO the caller's buffer, the argument
    /// errors come in the documented order, and an absent field leaves the out-params alone.
    #[test]
    fn parse_response_header_points_into_the_callers_buffer() {
        let mut r = Run::new();
        let h = r.cstr(0x100, "Content-Length: 42\r\nServer: x\r\n");
        let name = r.cstr(0x40, "content-length");
        let (vp, lp) = (0x80, 0x84);
        assert_eq!(r.call(parse_response_header, &[h, 31, name, vp, lp]), 20);
        assert_eq!(
            u32::from_le_bytes(r.bytes[vp as usize..vp as usize + 4].try_into().unwrap()),
            h + 16
        );
        assert_eq!(
            u32::from_le_bytes(r.bytes[lp as usize..lp as usize + 4].try_into().unwrap()),
            2
        );
        assert_eq!(
            r.call(parse_response_header, &[0, 31, name, vp, lp]),
            SCE_HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE
        );
        assert_eq!(
            r.call(parse_response_header, &[h, 31, name, 0, lp]),
            SCE_HTTP_ERROR_PARSE_HTTP_INVALID_VALUE
        );
        assert_eq!(
            r.call(parse_response_header, &[h, 0, name, vp, lp]),
            SCE_HTTP_ERROR_PARSE_HTTP_NOT_FOUND
        );
        let other = r.cstr(0x60, "etag");
        r.bytes[vp as usize..vp as usize + 4].copy_from_slice(&7u32.to_le_bytes());
        assert_eq!(
            r.call(parse_response_header, &[h, 31, other, vp, lp]),
            SCE_HTTP_ERROR_PARSE_HTTP_NOT_FOUND
        );
        assert_eq!(r.bytes[vp as usize], 7);
    }

    fn el(scheme: &str, host: &str, port: u16, path: &str) -> UriElement {
        UriElement {
            scheme: scheme.into(),
            hostname: host.into(),
            port,
            path: path.into(),
            ..Default::default()
        }
    }

    /// The default port is left out, an explicit one kept; opaque URIs have no `//`; userinfo,
    /// query and fragment appear only with their bits.
    #[test]
    fn a_uri_is_built_from_its_parts() {
        assert_eq!(
            build_uri(&el("http", "example.com", 8080, "/foo"), 0xFFFF),
            b"http://example.com:8080/foo"
        );
        assert_eq!(
            build_uri(&el("https", "a.b", 443, "/x"), 0xFFFF),
            b"https://a.b/x"
        );
        assert_eq!(
            build_uri(&el("file", "", 0, "/etc/passwd"), 0xFFFF),
            b"file:///etc/passwd"
        );
        let mut m = el("mailto", "", 0, "user@example.com");
        m.opaque = true;
        assert_eq!(build_uri(&m, 0xFFFF), b"mailto:user@example.com");
        let mut full = el("http", "h", 81, "/p");
        full.username = b"u".to_vec();
        full.password = b"pw".to_vec();
        full.query = b"?q=1".to_vec();
        full.fragment = b"#f".to_vec();
        assert_eq!(build_uri(&full, 0xFFFF), b"http://u:pw@h:81/p?q=1#f");
        assert_eq!(build_uri(&full, URI_HOSTNAME | URI_PATH), b"//h/p");
    }

    /// The two-call protocol through the guest entry point: a size query, a short buffer that
    /// is refused (with the size still reported), then the string with its NUL.
    #[test]
    fn uri_build_follows_the_two_call_protocol() {
        let mut r = Run::new();
        let scheme = r.cstr(0x40, "http");
        let host = r.cstr(0x50, "h");
        let path = r.cstr(0x60, "/p");
        let e = 0x100usize;
        r.bytes[e + 0x04..e + 0x08].copy_from_slice(&scheme.to_le_bytes());
        r.bytes[e + 0x10..e + 0x14].copy_from_slice(&host.to_le_bytes());
        r.bytes[e + 0x14..e + 0x18].copy_from_slice(&path.to_le_bytes());
        r.bytes[e + 0x20..e + 0x22].copy_from_slice(&8080u16.to_le_bytes());
        let req = 0x80u32;
        assert_eq!(r.call(uri_build, &[0, req, 0, e as u32, 0xFFFF]), 0);
        let need = u32::from_le_bytes(r.bytes[req as usize..req as usize + 4].try_into().unwrap());
        assert_eq!(need as usize, "http://h:8080/p".len() + 1);
        assert_eq!(
            r.call(uri_build, &[0x200, req, need - 1, e as u32, 0xFFFF]),
            SCE_HTTP_ERROR_OUT_OF_MEMORY
        );
        assert_eq!(r.bytes[0x200], 0);
        assert_eq!(r.call(uri_build, &[0x200, req, need, e as u32, 0xFFFF]), 0);
        assert_eq!(&r.bytes[0x200..0x200 + need as usize], b"http://h:8080/p\0");
        assert_eq!(
            r.call(uri_build, &[0x200, req, need, 0, 0xFFFF]),
            SCE_HTTP_ERROR_INVALID_URL
        );
        assert_eq!(
            r.call(uri_build, &[0, 0, 0, e as u32, 0xFFFF]),
            SCE_HTTP_ERROR_INVALID_VALUE
        );
    }

    /// Scheme and method are checked; a request with no response has no headers to give.
    #[test]
    fn scheme_method_and_header_block_are_checked() {
        let mut r = Run::new();
        let ua = r.cstr(0x10, "agent");
        let tmpl = r.call(create_template, &[ua, 2, 0]);
        let server = r.cstr(0x40, "h");
        let ftp = r.cstr(0x80, "ftp");
        assert_eq!(
            r.call(create_connection, &[tmpl as u32, server, ftp, 21, 0]),
            SCE_HTTP_ERROR_UNKNOWN_SCHEME
        );
        let upper = r.cstr(0x90, "HTTP");
        let conn = r.call(create_connection, &[tmpl as u32, server, upper, 0, 0]);
        assert!(conn > 0);
        let path = r.cstr(0xC0, "/");
        assert_eq!(
            r.call(create_request, &[conn as u32, 8, path]),
            SCE_HTTP_ERROR_UNKNOWN_METHOD
        );
        assert_eq!(
            r.call(create_request, &[conn as u32, 8, 0]),
            SCE_HTTP_ERROR_UNKNOWN_METHOD
        );
        let req = r.call(create_request, &[conn as u32, 0, path]);
        assert!(req > 0);
        assert_eq!(
            r.call(get_all_response_headers, &[req as u32, 0x200, 0x204]),
            SCE_HTTP_ERROR_BEFORE_SEND
        );
        assert_eq!(
            r.call(get_all_response_headers, &[req as u32, 0, 0x204]),
            SCE_HTTP_ERROR_INVALID_VALUE
        );
        assert_eq!(
            r.call(get_all_response_headers, &[0x999, 0x200, 0x204]),
            SCE_HTTP_ERROR_INVALID_ID
        );
    }

    /// The auth and redirect hooks are accepted on any live object - a template, as titles set
    /// them before any connection exists - and refused on an id that names nothing.
    #[test]
    fn response_callbacks_need_a_live_object() {
        let mut r = Run::new();
        let ua = r.cstr(0x10, "agent");
        let tmpl = r.call(create_template, &[ua, 2, 0]);
        assert_eq!(r.call(set_response_callback, &[tmpl as u32, 0, 0]), 0);
        assert_eq!(
            r.call(set_response_callback, &[tmpl as u32, 0x1234, 0x10]),
            0
        );
        assert_eq!(
            r.call(set_response_callback, &[0x7777, 0, 0]),
            SCE_HTTP_ERROR_INVALID_ID
        );
        assert_eq!(r.call(delete_template, &[tmpl as u32]), 0);
        assert_eq!(
            r.call(set_response_callback, &[tmpl as u32, 0, 0]),
            SCE_HTTP_ERROR_INVALID_ID
        );
    }
}
