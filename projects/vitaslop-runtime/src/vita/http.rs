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

use crate::host::{GuestCtx, HttpKind, VitaState};
use crate::hostcall;

/// `SceHttpErrorCode` values this surface uses, from `psp2/net/http.h`.
const SCE_HTTP_ERROR_INVALID_ID: i32 = 0x8043_1100u32 as i32;
const SCE_HTTP_ERROR_INVALID_VALUE: i32 = 0x8043_11FEu32 as i32;
const SCE_HTTP_ERROR_BEFORE_SEND: i32 = 0x8043_1065u32 as i32;
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
    let ua = if user_agent.is_null() { String::new() } else { ctx.read_cstr(user_agent.addr(), LABEL_MAX) };
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
    let r = if st.http_set_timeout(id, which, usec) { 0 } else { SCE_HTTP_ERROR_INVALID_ID };
    ctx.ret(r as u32);
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
        let value = if value.is_null() { String::new() } else { ctx.read_cstr(value.addr(), LABEL_MAX) };
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
pub(super) fn get_status_code(ctx: &mut GuestCtx, st: &mut VitaState, req: i32, status: Ptr) -> i32 {
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
pub(super) fn get_response_content_length(ctx: &mut GuestCtx, st: &mut VitaState, req: i32, len: Ptr) -> i32 {
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
pub(super) fn ssl_get_ssl_error(ctx: &mut GuestCtx, st: &mut VitaState, id: i32, err_num: Ptr, detail: Ptr) -> i32 {
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
