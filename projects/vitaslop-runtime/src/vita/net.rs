//! SceNet: the BSD-sockets surface, modelled OFFLINE.
//!
//! # Why offline, and what "offline" means here
//! Nothing in this emulator opens a host socket. That is a deliberate boundary, not a
//! missing feature: a run has to be deterministic and reproducible from the same
//! inputs, and a real network is neither. It is also the only honest position - a title
//! handed a fabricated server reply would act on data nobody sent.
//!
//! So the model is a console **with the network interface down**. That is a real,
//! coherent state the console has, and every title ships a path for it: every call
//! below either does its work purely locally (byte order, address parsing, the socket
//! table) or reports exactly the errno a down interface produces. Nothing here returns
//! a hollow success.
//!
//! # What IS real
//! - The socket table: `sceNetSocket` really allocates a descriptor, `sceNetBind`
//!   really records the local address, `sceNetGetsockname` reads it back,
//!   `sceNetSetsockopt`/`Getsockopt` really round-trip an option, and
//!   `sceNetSocketClose` really frees it. A title that manages descriptors correctly
//!   sees correct behaviour; one that leaks them runs out.
//! - The pure conversions - `sceNetHtonl`/`Htons`/`Ntohl`/`Ntohs`, `sceNetInetPton`/
//!   `InetNtop` - are exact. They touch no network at all, and a title uses them to
//!   build strings it prints.
//! - `sceNetErrnoLoc` hands back a per-thread errno slot in guest memory, and every
//!   failing call below sets it, because that is where a title reads WHY.
//!
//! # What reports the interface being down
//! Connect, send, receive, name resolution, and anything that needs a peer. The errno
//! is `SCE_NET_ENETDOWN` (or `ENOTCONN` where the socket state is what is wrong, which
//! is more specific and equally true), matching `sceNetCtlInetGetState` reporting
//! disconnected - the two must agree, or a title that checks the link state first and
//! then gets a different story from a socket call is being told two different things.

use crate::host::{GuestCtx, Ptr, VitaState};
use crate::hostcall;

/// `SceNetErrorCode` values this surface uses. Positive BSD errno values, which is how
/// SceNet reports them through `sceNetErrnoLoc`; the CALL returns -1.
pub(super) const SCE_NET_ENOTCONN: i32 = 128;
pub(super) const SCE_NET_ENETDOWN: i32 = 116;
pub(super) const SCE_NET_EBADF: i32 = 9;
pub(super) const SCE_NET_EINVAL: i32 = 22;
pub(super) const SCE_NET_EAGAIN: i32 = 11;
/// `SCE_NET_RESOLVER_ERROR_NO_RECORD` - the resolver found no record for the name.
const SCE_NET_RESOLVER_ENODNS: i32 = 0x804_1140Au32 as i32;

/// What every failing socket call returns; the reason goes in the thread's errno.
const NET_FAIL: i32 = -1;

/// Fail `code`: record it in the calling thread's errno slot and return -1.
fn fail(st: &mut VitaState, code: i32) -> i32 {
    st.net_set_errno(code);
    NET_FAIL
}

/// int sceNetSocket(const char *name, int domain, int type, int protocol)
///
/// Really allocates a descriptor. A socket can be created on a console whose link is
/// down - it is only the operations that need the network that fail - so refusing here
/// would be wrong, and would also hide which call a title actually cannot complete.
#[hostcall]
pub(super) fn socket(ctx: &mut GuestCtx, st: &mut VitaState, name: Ptr, domain: i32, ty: i32, protocol: i32) -> i32 {
    let name = if name.is_null() { String::new() } else { ctx.read_cstr(name.addr(), 64) };
    st.net_socket(&name, domain, ty, protocol)
}

/// int sceNetSocketClose(int s)
#[hostcall]
pub(super) fn socket_close(st: &mut VitaState, s: i32) -> i32 {
    if st.net_close(s) {
        0
    } else {
        fail(st, SCE_NET_EBADF)
    }
}

/// `SceNetSockaddrIn`: `{ u8 len, u8 family, u16 port (network order), u32 addr
/// (network order), u8 zero[6] }`, 16 bytes - the same shape as BSD `sockaddr_in`.
const SOCKADDR_LEN: usize = 16;
const SOCKADDR_PORT: u32 = 2;
const SOCKADDR_ADDR: u32 = 4;

/// int sceNetBind(int s, const SceNetSockaddr *addr, unsigned int addrlen)
///
/// Binding names a LOCAL endpoint, which needs no network, so this really records the
/// address and `sceNetGetsockname` reads it back.
#[hostcall]
pub(super) fn bind(ctx: &mut GuestCtx, st: &mut VitaState, s: i32, addr: Ptr, _addrlen: u32) -> i32 {
    if addr.is_null() {
        fail(st, SCE_NET_EINVAL)
    } else {
        let port = ctx.read_u32(addr.addr() + SOCKADDR_PORT) as u16;
        let ip = ctx.read_u32(addr.addr() + SOCKADDR_ADDR);
        if st.net_bind(s, ip, port) {
            0
        } else {
            fail(st, SCE_NET_EBADF)
        }
    }
}

/// int sceNetListen(int s, int backlog)
/// A listening socket is local state; it is `sceNetAccept` that needs a peer.
#[hostcall]
pub(super) fn listen(st: &mut VitaState, s: i32, _backlog: i32) -> i32 {
    if st.net_listen(s) {
        0
    } else {
        fail(st, SCE_NET_EBADF)
    }
}

/// int sceNetAccept(int s, SceNetSockaddr *addr, unsigned int *addrlen)
///
/// No peer can reach a console whose link is down, so there is never a pending
/// connection. `EAGAIN` is the right answer rather than an error: it is what a
/// non-blocking accept with an empty queue returns, and a server loop handles it by
/// going round again instead of tearing itself down.
#[hostcall]
pub(super) fn accept(st: &mut VitaState, _s: i32, _addr: Ptr, _addrlen: Ptr) -> i32 {
    fail(st, SCE_NET_EAGAIN)
}

/// int sceNetConnect(int s, const SceNetSockaddr *name, unsigned int namelen)
#[hostcall]
pub(super) fn connect(st: &mut VitaState, _s: i32, _name: Ptr, _namelen: u32) -> i32 {
    fail(st, SCE_NET_ENETDOWN)
}

/// int sceNetSend / sceNetSendto / sceNetSendmsg
///
/// `ENOTCONN` rather than `ENETDOWN`: nothing ever connects (see `connect`), so the
/// socket's own state is the nearer truth and it is the errno a title's send path
/// expects to see after a failed connect.
#[hostcall]
pub(super) fn send(st: &mut VitaState, _s: i32) -> i32 {
    fail(st, SCE_NET_ENOTCONN)
}

/// int sceNetRecv / sceNetRecvfrom
#[hostcall]
pub(super) fn recv(st: &mut VitaState, _s: i32) -> i32 {
    fail(st, SCE_NET_ENOTCONN)
}

/// int sceNetShutdown(int s, int how)
#[hostcall]
pub(super) fn shutdown(st: &mut VitaState, s: i32, _how: i32) -> i32 {
    if st.net_socket_exists(s) {
        0
    } else {
        fail(st, SCE_NET_EBADF)
    }
}

/// int sceNetSocketAbort(int s, int flags)
///
/// Unblock everything waiting on a socket, so a worker parked in a receive can be torn
/// down. Nothing here ever blocks on a socket - every call that would need a peer fails at
/// once with the link down - so there is never a wait for this to break, and the abort has
/// genuinely done all there is to do. The descriptor is still CHECKED, because aborting one
/// that does not exist is a caller error the console reports and this engine's socket table
/// can see.
///
/// `flags` selects which directions to abort (send, receive, or both). With no blocked
/// operation in either direction the selection changes nothing, which is why it is not read.
#[hostcall]
pub(super) fn socket_abort(st: &mut VitaState, s: i32, _flags: i32) -> i32 {
    if st.net_socket_exists(s) {
        0
    } else {
        fail(st, SCE_NET_EBADF)
    }
}

/// int sceNetGetsockname(int s, SceNetSockaddr *name, unsigned int *namelen)
///
/// Reads back what `sceNetBind` recorded. An unbound socket reports the wildcard
/// address and port zero, which is what the kernel reports for one.
#[hostcall]
pub(super) fn getsockname(ctx: &mut GuestCtx, st: &mut VitaState, s: i32, name: Ptr, namelen: Ptr) -> i32 {
    match st.net_local_addr(s) {
        None => fail(st, SCE_NET_EBADF),
        Some((ip, port)) => {
            write_sockaddr(ctx, name, namelen, ip, port);
            0
        }
    }
}

/// int sceNetGetpeername(int s, SceNetSockaddr *name, unsigned int *namelen)
/// Nothing is ever connected, so there is no peer to name.
#[hostcall]
pub(super) fn getpeername(st: &mut VitaState, _s: i32, _name: Ptr, _namelen: Ptr) -> i32 {
    fail(st, SCE_NET_ENOTCONN)
}

/// Fill a `SceNetSockaddrIn` and its length out-parameter.
fn write_sockaddr(ctx: &mut GuestCtx, addr: Ptr, addrlen: Ptr, ip: u32, port: u16) {
    if !addr.is_null() {
        let mut buf = [0u8; SOCKADDR_LEN];
        buf[0] = SOCKADDR_LEN as u8;
        buf[1] = 2; // SCE_NET_AF_INET
        buf[2..4].copy_from_slice(&port.to_be_bytes());
        buf[4..8].copy_from_slice(&ip.to_le_bytes());
        ctx.write_bytes(addr.addr(), &buf);
    }
    if !addrlen.is_null() {
        ctx.write_u32(addrlen.addr(), SOCKADDR_LEN as u32);
    }
}

/// int sceNetSetsockopt(int s, int level, int optname, const void *optval, unsigned int optlen)
///
/// Options are per-socket state a title sets and reads back (a send timeout, a buffer
/// size, `SO_REUSEADDR`), and none of them need the network, so they really round-trip.
#[hostcall]
pub(super) fn setsockopt(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    s: i32,
    level: i32,
    optname: i32,
    optval: Ptr,
    optlen: u32,
) -> i32 {
    if optval.is_null() || optlen < 4 {
        fail(st, SCE_NET_EINVAL)
    } else {
        let v = ctx.read_u32(optval.addr());
        if st.net_set_opt(s, level, optname, v) {
            0
        } else {
            fail(st, SCE_NET_EBADF)
        }
    }
}

/// int sceNetGetsockopt(int s, int level, int optname, void *optval, unsigned int *optlen)
///
/// An option that was never set reads back as zero, which is the kernel's own default
/// for every option a title queries here.
#[hostcall]
pub(super) fn getsockopt(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    s: i32,
    level: i32,
    optname: i32,
    optval: Ptr,
    optlen: Ptr,
) -> i32 {
    match st.net_get_opt(s, level, optname) {
        None => fail(st, SCE_NET_EBADF),
        Some(v) => {
            if !optval.is_null() {
                ctx.write_u32(optval.addr(), v);
            }
            if !optlen.is_null() {
                ctx.write_u32(optlen.addr(), 4);
            }
            0
        }
    }
}

/// int sceNetGetSockInfo(int s, SceNetSockInfo *info, int n, int flags)
///
/// A diagnostic dump of the socket's kernel state. The state we model - descriptor,
/// bound address - is real; there is no connection state to report because there are no
/// connections. Reports zero records rather than one with invented contents.
#[hostcall]
pub(super) fn get_sock_info(_st: &mut VitaState, _s: i32) -> i32 {
    0
}

/// int sceNetShowNetstat(void)
///
/// Prints the socket table to the debug console, which is exactly what it does on a
/// console - and here the table is real, so the output is genuinely informative.
pub(super) fn show_netstat(ctx: &mut GuestCtx, st: &mut VitaState) {
    let dump = st.net_netstat();
    st.write_stdout(dump.as_bytes());
    ctx.ret(0);
}

// --- Byte order and address conversion: pure, and exactly right ---------------

/// unsigned int sceNetHtonl(unsigned int n) / sceNetNtohl(unsigned int n)
#[hostcall]
pub(super) fn swap32(n: u32) -> u32 {
    n.swap_bytes()
}

/// unsigned short sceNetHtons(unsigned short n) / sceNetNtohs(unsigned short n)
#[hostcall]
pub(super) fn swap16(n: u32) -> u32 {
    ((n as u16).swap_bytes()) as u32
}

/// int sceNetInetPton(int af, const char *src, void *dst)
///
/// Parse a dotted-quad into a network-order address. Returns 1 on success, 0 on a
/// malformed string - the BSD convention, which is NOT the -1/errno convention the rest
/// of this surface uses, and a title tests it as a boolean.
#[hostcall]
pub(super) fn inet_pton(ctx: &mut GuestCtx, st: &mut VitaState, af: i32, src: Ptr, dst: Ptr) -> i32 {
    if af != 2 || src.is_null() || dst.is_null() {
        fail(st, SCE_NET_EINVAL)
    } else {
        let text = ctx.read_cstr(src.addr(), 64);
        match parse_ipv4(&text) {
            Some(be) => {
                ctx.write_u32(dst.addr(), be);
                1
            }
            None => 0,
        }
    }
}

/// const char *sceNetInetNtop(int af, const void *src, char *dst, unsigned int size)
///
/// Format a network-order address as a dotted quad into the caller's buffer, returning
/// that buffer (or null on failure), which is what the caller then prints.
#[hostcall]
pub(super) fn inet_ntop(ctx: &mut GuestCtx, st: &mut VitaState, af: i32, src: Ptr, dst: Ptr, size: u32) -> u32 {
    if af != 2 || src.is_null() || dst.is_null() || size < 8 {
        st.net_set_errno(SCE_NET_EINVAL);
        0
    } else {
        let be = ctx.read_u32(src.addr());
        let b = be.to_le_bytes();
        let text = format!("{}.{}.{}.{}", b[0], b[1], b[2], b[3]);
        let mut buf = text.into_bytes();
        buf.truncate(size as usize - 1);
        buf.push(0);
        ctx.write_bytes(dst.addr(), &buf);
        dst.addr()
    }
}

/// Parse `a.b.c.d` into the 4 bytes in network order (which, read as a little-endian
/// u32, puts `a` in the low byte - the same layout `inet_ntop` reads back).
fn parse_ipv4(text: &str) -> Option<u32> {
    let mut bytes = [0u8; 4];
    let mut parts = text.trim().split('.');
    for b in bytes.iter_mut() {
        *b = parts.next()?.parse::<u8>().ok()?;
    }
    if parts.next().is_some() {
        return None;
    }
    Some(u32::from_le_bytes(bytes))
}

// --- Name resolution ----------------------------------------------------------

/// int sceNetResolverCreate(const char *name, void *param, int flags)
///
/// A resolver handle is local, so it is really created; it is resolving that cannot
/// work. Creating it lets a title's teardown path (`Destroy`) stay balanced.
#[hostcall]
pub(super) fn resolver_create(st: &mut VitaState, _name: Ptr, _param: Ptr, _flags: i32) -> i32 {
    st.net_resolver_create()
}

/// int sceNetResolverAbort(int rid, int flags)
///
/// Cancels an in-flight resolve. Nothing here is ever in flight - `resolver_start`
/// fails synchronously with no record - so there is never anything to cancel, and
/// saying so is the truthful answer for a live handle. A handle that does not exist is
/// still `EBADF`, which is the distinction a title's teardown path cares about.
#[hostcall]
pub(super) fn resolver_abort(st: &mut VitaState, rid: i32, _flags: i32) -> i32 {
    if st.net_resolver_error(rid).is_some() { 0 } else { fail(st, SCE_NET_EBADF) }
}

/// int sceNetGetMacAddress(SceNetEtherAddr *addr, int flags)
///
/// The interface is down, not absent: a Vita has a MAC address whether or not it is
/// associated, and a title uses it as a stable per-console identity (a seed, a save
/// tag, a peer name). So this succeeds with a FIXED, locally-administered address -
/// the `0x02` bit in the first octet is what "not globally unique, assigned locally"
/// means in IEEE 802, which is exactly what this is. It is the same every run, because
/// a title that stores it must find it again.
///
/// It is deliberately not a random or host-derived address: a real one would leak the
/// player's machine identity into a save file, and a random one would make a title that
/// remembers it think it moved to a different console on every launch.
#[hostcall]
pub(super) fn get_mac_address(ctx: &mut GuestCtx, st: &mut VitaState, addr: Ptr, _flags: i32) -> i32 {
    if addr.is_null() {
        fail(st, SCE_NET_EINVAL)
    } else {
        ctx.write_bytes(addr.addr(), &[0x02, 0x00, 0x00, 0x76, 0x69, 0x74]);
        0
    }
}

/// int sceNetShowIfconfig(void *p, int b)
///
/// A DIAGNOSTIC dump of the interface configuration, which the real one prints to the
/// system log. There is no interface to describe and nothing that reads our stdout as
/// an ifconfig table, so it does its work by reporting the state once and succeeding.
#[hostcall]
pub(super) fn show_ifconfig(st: &mut VitaState, _p: Ptr, _b: i32) -> i32 {
    tracing::info!(
        target: "vitaslop::status",
        thread = st.current_thread(),
        "sceNetShowIfconfig: no interface is up (see `vita::net` - the network is modelled down)"
    );
    0
}

/// int sceNetDumpCreate(const char *name, int len, int flags)
///
/// A packet-capture handle over an interface. There is no interface, so there is
/// nothing to capture from and no handle to give: `ENETDOWN` is the same answer every
/// other call that needs the link returns, and a title is told one consistent story.
#[hostcall]
pub(super) fn dump_create(st: &mut VitaState, _name: Ptr, _len: i32, _flags: i32) -> i32 {
    fail(st, SCE_NET_ENETDOWN)
}

/// int sceNetDumpDestroy(int id)
///
/// No dump handle was ever created, so any id is a bad one.
#[hostcall]
pub(super) fn dump_destroy(st: &mut VitaState, _id: i32) -> i32 {
    fail(st, SCE_NET_EBADF)
}

/// int sceNetDumpRead(int id, void *buf, int len, int *pflags)
#[hostcall]
pub(super) fn dump_read(st: &mut VitaState, _id: i32, _buf: Ptr, _len: i32, _pflags: Ptr) -> i32 {
    fail(st, SCE_NET_EBADF)
}

/// int sceNetResolverDestroy(int rid)
#[hostcall]
pub(super) fn resolver_destroy(st: &mut VitaState, rid: i32) -> i32 {
    if st.net_resolver_destroy(rid) {
        0
    } else {
        fail(st, SCE_NET_EBADF)
    }
}

/// int sceNetResolverStartNtoa(int rid, const char *hostname, SceNetInAddr *addr,
///     int timeout, int retry, int flags) - and the Aton (reverse) direction.
///
/// There is no DNS server to ask, so no name has a record. `NO_RECORD` is the truthful
/// answer and the one a title's "cannot reach the server" path is written for; a
/// fabricated address would send it on to connect to a machine that does not exist and
/// fail later, further from the cause.
#[hostcall]
pub(super) fn resolver_start(st: &mut VitaState, rid: i32) -> i32 {
    st.net_resolver_set_error(rid, SCE_NET_RESOLVER_ENODNS);
    SCE_NET_RESOLVER_ENODNS
}

/// int sceNetResolverGetError(int rid, int *result)
#[hostcall]
pub(super) fn resolver_get_error(ctx: &mut GuestCtx, st: &mut VitaState, rid: i32, result: Ptr) -> i32 {
    match st.net_resolver_error(rid) {
        None => fail(st, SCE_NET_EBADF),
        Some(e) => {
            if !result.is_null() {
                ctx.write_u32(result.addr(), e as u32);
            }
            0
        }
    }
}

// --- epoll --------------------------------------------------------------------

/// int sceNetEpollCreate(const char *name, int flags)
#[hostcall]
pub(super) fn epoll_create(st: &mut VitaState, _name: Ptr, _flags: i32) -> i32 {
    st.net_epoll_create()
}

/// int sceNetEpollDestroy(int eid)
#[hostcall]
pub(super) fn epoll_destroy(st: &mut VitaState, eid: i32) -> i32 {
    if st.net_epoll_destroy(eid) {
        0
    } else {
        fail(st, SCE_NET_EBADF)
    }
}

/// int sceNetEpollControl(int eid, int op, int id, SceNetEpollEvent *event)
///
/// Registering interest is local bookkeeping and really happens, so a title's add /
/// modify / delete sequence stays consistent with what `sceNetEpollWait` then reports.
#[hostcall]
pub(super) fn epoll_control(st: &mut VitaState, eid: i32, op: i32, id: i32, _event: Ptr) -> i32 {
    if st.net_epoll_control(eid, op, id) {
        0
    } else {
        fail(st, SCE_NET_EBADF)
    }
}

/// int sceNetEpollWait(int eid, SceNetEpollEvent *events, int maxevents, int timeout)
///
/// No socket can ever become readable or writable (nothing connects), so the wait
/// always expires with ZERO events ready. Returning 0 is the timeout answer, not an
/// error, and it is what keeps a poll loop looping instead of aborting.
#[hostcall]
pub(super) fn epoll_wait(st: &mut VitaState, eid: i32, _events: Ptr, _maxevents: i32, _timeout: i32) -> i32 {
    if st.net_epoll_exists(eid) {
        0
    } else {
        fail(st, SCE_NET_EBADF)
    }
}

// --- SceNetAdhocMatching -----------------------------------------------------------
//
// Peer discovery over the ad-hoc radio: a title creates a matching context as PARENT or
// CHILD, starts it, and is called back as consoles appear and offer to join.
//
// >>> THE MODEL IS "THE RADIO WORKS AND NOBODY IS THERE", not "the library is broken".
// That is a real state a console is in constantly - the first player to open a lobby sits
// in exactly it - and it is the same stance `sceNetCtlAdhocGetPeerList` already takes by
// reporting an EMPTY list rather than an error. So the lifecycle calls all succeed: a
// context is created, started, stopped and deleted for real, and the handler callback is
// simply never invoked, because no peer ever appears to invoke it about. A title that
// waits for peers waits, which is what it does on hardware in an empty room; a title that
// was refused at `Create` would instead show an error it need not show.
//
// The ids are minted in creation order, so they are a function of the guest's own call
// sequence and identical across runs (see `VitaState::adhoc_matching_create`).

/// `SceNetAdhocMatchingErrorCode` values this surface reports.
const SCE_NET_ADHOC_MATCHING_ERROR_INVALID_MODE: i32 = 0x8041_3101u32 as i32;
const SCE_NET_ADHOC_MATCHING_ERROR_INVALID_MAXNUM: i32 = 0x8041_3103u32 as i32;
const SCE_NET_ADHOC_MATCHING_ERROR_INVALID_ID: i32 = 0x8041_3107u32 as i32;
const SCE_NET_ADHOC_MATCHING_ERROR_UNKNOWN_TARGET: i32 = 0x8041_310Cu32 as i32;
const SCE_NET_ADHOC_MATCHING_ERROR_ALREADY_INITIALIZED: i32 = 0x8041_3112u32 as i32;
const SCE_NET_ADHOC_MATCHING_ERROR_NOT_INITIALIZED: i32 = 0x8041_3113u32 as i32;

/// int sceNetAdhocMatchingInit(unsigned int pool_size, void *pool_ptr)
///
/// The library is given a memory pool by the CALLER - so there is nothing to allocate
/// here, only the fact of initialisation to record, which is what lets a `Create` before
/// it report NOT_INITIALIZED instead of quietly working.
#[hostcall]
pub(super) fn adhoc_matching_init(st: &mut VitaState, pool_size: u32, pool_ptr: u32) -> i32 {
    if st.adhoc_matching_init(pool_ptr, pool_size) {
        0
    } else {
        SCE_NET_ADHOC_MATCHING_ERROR_ALREADY_INITIALIZED
    }
}

/// int sceNetAdhocMatchingCreate(SceNetAdhocMatchingMode mode, int max_members,
///     SceUShort16 port, int rx_buffer_len, unsigned int hello_interval,
///     unsigned int keep_alive_interval, int retry_count, unsigned int rexmt_interval,
///     SceNetAdhocMatchingCallback handler)
///
/// Returns the context id, or an error. `mode` is PARENT(1) / CHILD(2) / P2P(3) and
/// anything else is rejected, as is a non-positive member count: those are the two
/// validations a caller can trip by itself, and they are worth keeping because they are
/// the same answer the console gives.
///
/// The `handler` is deliberately not stored. Nothing can ever call it - see the note
/// above - and holding a callback that is never invoked would suggest otherwise.
#[hostcall]
#[allow(clippy::too_many_arguments)]
pub(super) fn adhoc_matching_create(
    st: &mut VitaState,
    mode: i32,
    max_members: i32,
    _port: u32,
    _rx_buffer_len: i32,
    _hello_interval: u32,
    _keep_alive_interval: u32,
    _retry_count: i32,
    _rexmt_interval: u32,
    _handler: u32,
) -> i32 {
    if !st.adhoc_matching_ready() {
        SCE_NET_ADHOC_MATCHING_ERROR_NOT_INITIALIZED
    } else if !(1..=3).contains(&mode) {
        SCE_NET_ADHOC_MATCHING_ERROR_INVALID_MODE
    } else if max_members <= 0 {
        SCE_NET_ADHOC_MATCHING_ERROR_INVALID_MAXNUM
    } else {
        st.adhoc_matching_create()
    }
}

/// int sceNetAdhocMatchingStart(int id, int thread_priority, int thread_stack_size,
///     int thread_cpu_affinity_mask, int hello_opt_len, void *hello_opt)
/// int sceNetAdhocMatchingStop(int id)
///
/// Start begins advertising and listening. Both really move the context's state, so a
/// double start or a stop of an id that was never created is reported rather than
/// accepted. `start` selects which.
pub(super) fn adhoc_matching_set_started(ctx: &mut GuestCtx, st: &mut VitaState, start: bool) {
    let id = ctx.arg(0) as i32;
    let r = match st.adhoc_matching_set_started(id, start) {
        Some(()) => 0,
        None => SCE_NET_ADHOC_MATCHING_ERROR_INVALID_ID,
    };
    ctx.ret(r as u32);
}

/// int sceNetAdhocMatchingDelete(int id)
#[hostcall]
pub(super) fn adhoc_matching_delete(st: &mut VitaState, id: i32) -> i32 {
    if st.adhoc_matching_delete(id) {
        0
    } else {
        SCE_NET_ADHOC_MATCHING_ERROR_INVALID_ID
    }
}

/// int sceNetAdhocMatchingTerm(void)
///
/// Hand the pool back and drop every context with it (see
/// [`VitaState::adhoc_matching_term`] for why the contexts go too). A term before an init is
/// the one way this can fail, and it is a caller error the console reports.
#[hostcall]
pub(super) fn adhoc_matching_term(st: &mut VitaState) -> i32 {
    if st.adhoc_matching_term() {
        tracing::info!(
            target: "vitaslop::status",
            thread = st.current_thread(),
            "SceNetAdhocMatching: no wireless link and no nearby consoles - matchmaking found no peers"
        );
        0
    } else {
        SCE_NET_ADHOC_MATCHING_ERROR_NOT_INITIALIZED
    }
}

/// int sceNetAdhocMatchingSendData(int id, SceNetInAddr *addr, int data_len, void *data)
///
/// Send a payload to one paired peer. It gets the same answer
/// [`adhoc_matching_select_target`] gives, for the same reason: no handler has ever fired on
/// this context, so no address the title can pass is one it has heard from, and
/// UNKNOWN_TARGET is what the console says about an address that is not in its member list.
///
/// Reporting success would be a claim that bytes went out to somebody - the one answer that
/// cannot be true here, and one a title acts on by waiting for a reply that cannot come.
#[hostcall]
pub(super) fn adhoc_matching_send_data(st: &mut VitaState, id: i32, _addr: Ptr, _data_len: i32, _data: Ptr) -> i32 {
    if st.adhoc_matching_live(id) {
        SCE_NET_ADHOC_MATCHING_ERROR_UNKNOWN_TARGET
    } else {
        SCE_NET_ADHOC_MATCHING_ERROR_INVALID_ID
    }
}

/// int sceNetAdhocMatchingCancelTarget(int id, SceNetInAddr *target)
///
/// Withdraw a pairing request. Unlike [`adhoc_matching_send_data`] this SUCCEEDS on a live
/// context: it is a cancel, no request to that address is outstanding (none could be), and
/// so the state it asks for is the state already in force. A title reaches this from its own
/// cleanup path, and the codebase's rule for cancels holds here - there is nothing left
/// running to stop, so nothing can fail to stop. The context id is still checked, because a
/// cancel on a context that does not exist is a caller error the console reports.
#[hostcall]
pub(super) fn adhoc_matching_cancel_target(st: &mut VitaState, id: i32, _target: Ptr) -> i32 {
    if st.adhoc_matching_live(id) {
        0
    } else {
        SCE_NET_ADHOC_MATCHING_ERROR_INVALID_ID
    }
}

/// int sceNetAdhocMatchingGetMembers(int id, unsigned int *members_count,
///     struct SceNetAdhocMatchingMember *members)
///
/// ZERO members, and the count is written even when the buffer is null - that is the
/// two-call idiom this API is used with (ask the count, allocate, ask again), and a title
/// that skipped the write would allocate from a stale number. An empty member list is a real
/// state, not a failure: it is what the first console into a room sees.
#[hostcall]
pub(super) fn adhoc_matching_get_members(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    id: i32,
    count: Ptr,
    _members: Ptr,
) -> i32 {
    if !st.adhoc_matching_live(id) {
        SCE_NET_ADHOC_MATCHING_ERROR_INVALID_ID
    } else {
        if !count.is_null() {
            ctx.write_u32(count.addr(), 0);
        }
        0
    }
}

/// int sceNetAdhocMatchingSetHelloOpt(int id, int opt_len, void *opt)
///
/// The "hello" payload advertised to peers. There is nobody to advertise to, so the payload
/// is accepted and goes nowhere - which is what happens on a console in an empty room too:
/// the hello really is broadcast, and nothing answers.
#[hostcall]
pub(super) fn adhoc_matching_set_hello_opt(st: &mut VitaState, id: i32, _opt_len: i32, _opt: Ptr) -> i32 {
    if st.adhoc_matching_live(id) {
        0
    } else {
        SCE_NET_ADHOC_MATCHING_ERROR_INVALID_ID
    }
}

/// int sceNetAdhocMatchingSelectTarget(int id, SceNetInAddr *target, int opt_len,
///     void *opt)
///
/// Offer to pair with a peer the handler reported. No handler has ever fired here, so
/// whatever address the title passes names a console this context has never heard from -
/// UNKNOWN_TARGET, which is exactly what the console says about an address that is not in
/// its member list. Reporting success would promise a pairing that can never complete.
#[hostcall]
pub(super) fn adhoc_matching_select_target(st: &mut VitaState, id: i32, _target: Ptr, _opt_len: i32, _opt: Ptr) -> i32 {
    if st.adhoc_matching_live(id) {
        SCE_NET_ADHOC_MATCHING_ERROR_UNKNOWN_TARGET
    } else {
        SCE_NET_ADHOC_MATCHING_ERROR_INVALID_ID
    }
}

/// int *sceNetErrnoLoc(void)
///
/// The address of the calling thread's errno. Per THREAD, because two workers failing
/// concurrently must not read each other's reason - which is exactly why the real API
/// hands back a location rather than a value.
#[hostcall]
pub(super) fn errno_loc(st: &mut VitaState) -> u32 {
    st.net_errno_addr()
}
