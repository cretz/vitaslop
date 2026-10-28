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

use crate::SvcOutcome;
use crate::host::{GuestCtx, Ptr, VitaState};
use crate::hostcall;

/// `SceNetErrorCode` values this surface uses. Positive BSD errno values, which is how
/// SceNet reports them through `sceNetErrnoLoc`; the CALL returns -1.
pub(super) const SCE_NET_ENOTCONN: i32 = 128;
pub(super) const SCE_NET_ENETDOWN: i32 = 116;
pub(super) const SCE_NET_EBADF: i32 = 9;
pub(super) const SCE_NET_EINVAL: i32 = 22;
pub(super) const SCE_NET_EAGAIN: i32 = 11;
pub(super) const SCE_NET_EINTR: i32 = 4;
pub(super) const SCE_NET_ENOTBLK: i32 = 15;
/// `SCE_NET_EPOLL_ABORT_FLAG_PRESERVATION`: keep an abort that found no waiter for the next wait.
const EPOLL_ABORT_FLAG_PRESERVATION: i32 = 0x1;
/// `SCE_NET_RESOLVER_ERROR_NO_RECORD` - the resolver found no record for the name.
const SCE_NET_RESOLVER_ENODNS: i32 = 0x804_1140Au32 as i32;

/// What every failing socket call returns; the reason goes in the thread's errno.
pub(crate) const NET_FAIL: i32 = -1;

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
pub(super) fn socket(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    name: Ptr,
    domain: i32,
    ty: i32,
    protocol: i32,
) -> i32 {
    let name = if name.is_null() {
        String::new()
    } else {
        ctx.read_cstr(name.addr(), 64)
    };
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
pub(super) fn bind(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    s: i32,
    addr: Ptr,
    _addrlen: u32,
) -> i32 {
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
pub(super) fn getsockname(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    s: i32,
    name: Ptr,
    namelen: Ptr,
) -> i32 {
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

/// What the SceNet family was initialised with, kept on [`VitaState`] as `net_libs`: each
/// library's pool size while it is initialised (`None` otherwise), and SceLibRudp's objects.
#[derive(Default)]
pub struct NetLibs {
    /// `sceNetInit`'s `SceNetInitParam::size` - libnet's own memory pool.
    pub(crate) net_pool: Option<u32>,
    /// `sceHttpInit`'s pool - see `vita::http`.
    pub(crate) http_pool: Option<u32>,
    /// `sceSslInit`'s pool - see `vita::http`.
    pub(crate) ssl_pool: Option<u32>,
    /// SceLibRudp - see [`Rudp`].
    pub(crate) rudp: Rudp,
}

/// `SceNetInitParam`: `{ void *memory, int size, int flags }` (`psp2/net/net.h`, 0xC).
const NET_INIT_PARAM_SIZE_OFF: u32 = 4;

/// int sceNetInit(SceNetInitParam *param)
///
/// libnet runs out of a pool the CALLER provides, so initialising it is recording that pool -
/// which is what [`get_statistics_info`] reports back. Initialising succeeds with the link down,
/// as it does on a console; a missing parameter block is the caller's error.
///
/// A SECOND init while libnet is up is `SCE_NET_ERROR_EBUSY`, and the first pool stays. That
/// code is the one titles are written to expect: two retail titles' own init paths compare the
/// return against exactly `0x80410110` and carry on as initialised - which only means anything
/// if the console answers a repeat init with it. Titles routinely init libnet from two places
/// (a game layer and a middleware layer), so this is a path they take every boot.
#[hostcall]
pub(super) fn net_init(ctx: &mut GuestCtx, st: &mut VitaState, param: Ptr) -> i32 {
    if param.is_null() {
        SCE_NET_ERROR_EINVAL
    } else if st.net_libs.net_pool.is_some() {
        SCE_NET_ERROR_EBUSY
    } else {
        st.net_libs.net_pool = Some(ctx.read_u32(param.addr() + NET_INIT_PARAM_SIZE_OFF));
        0
    }
}

/// int sceNetTerm(void)
///
/// Releases the pool. Succeeds even with no init behind it, as the rest of the online stack's
/// teardown does (see the success group in `vita::dispatch`).
#[hostcall]
pub(super) fn net_term(st: &mut VitaState) -> i32 {
    st.net_libs.net_pool = None;
    0
}

/// `SCE_NET_ERROR_EINVAL` / `SCE_NET_ERROR_ENOTINIT` (`psp2common/net.h`): the CALL-return forms
/// of the errno values, which the non-socket calls return directly.
const SCE_NET_ERROR_EINVAL: i32 = 0x8041_0116u32 as i32;
const SCE_NET_ERROR_ENOTINIT: i32 = 0x8041_01C8u32 as i32;
const SCE_NET_ERROR_EBUSY: i32 = 0x8041_0110u32 as i32;

/// int sceNetGetStatisticsInfo(SceNetStatisticsInfo *info, int flags)
///
/// `SceNetStatisticsInfo` (`psp2common/net.h`, 0x18): `{ kernel_mem_free_size,
/// kernel_mem_free_min, packet_count, packet_qos_count, libnet_mem_free_size,
/// libnet_mem_free_min }`.
/// - The packet counters are 0: no packet has ever crossed an interface that is down.
/// - libnet's pool is the size [`net_init`] was handed, all of it free now and at its
///   low-water mark - this engine keeps libnet's state on the host and never takes a byte of
///   the guest's pool.
/// - The two kernel words are the stack's own 1 MiB heap, all free - see
///   [`NET_KERNEL_HEAP_BYTES`].
///
/// Before `sceNetInit` there is no libnet to describe: `SCE_NET_ERROR_ENOTINIT`.
#[hostcall]
pub(super) fn get_statistics_info(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    info: Ptr,
    flags: i32,
) -> i32 {
    match st.net_libs.net_pool {
        None => SCE_NET_ERROR_ENOTINIT,
        Some(_) if info.is_null() => SCE_NET_ERROR_EINVAL,
        Some(pool) => {
            // The kernel words are the stack's own heap (`NET_KERNEL_HEAP_BYTES`), all of it
            // free: this engine's stack allocates nothing there. A non-zero `flags` returns
            // those four kernel words byte-swapped.
            let k = |v: u32| if flags != 0 { v.swap_bytes() } else { v };
            let heap = NET_KERNEL_HEAP_BYTES;
            for (i, v) in [k(heap), k(heap), k(0), k(0), pool, pool].iter().enumerate() {
                ctx.write_u32(info.addr() + 4 * i as u32, *v);
            }
            0
        }
    }
}

/// The kernel network stack's private heap: one 1 MiB kernel block it allocates its packet
/// buffers, interfaces and sockets from. `kernel_mem_free_size` / `_min` report this heap's
/// free bytes, now and at their low-water mark. From the firmware's own stack as
/// reverse-engineered (no device reading): a console reports a little under this once it has
/// used the stack; one that never brought an interface up, which is this engine's model, has
/// all of it. Below 0x8000 free the stack reports out-of-memory, so a small stand-in here
/// would read as a starved network.
const NET_KERNEL_HEAP_BYTES: u32 = 0x10_0000;

/// int sceNetShowRoute(void)
///
/// A DIAGNOSTIC dump of the routing table, the sibling of `sceNetShowIfconfig` and reported
/// the same way: with no interface up there is no route, so it says so once and succeeds.
#[hostcall]
pub(super) fn show_route(st: &mut VitaState) -> i32 {
    tracing::info!(
        target: "vitaslop::status",
        thread = st.current_thread(),
        "sceNetShowRoute: no routes - no interface is up (see `vita::net`)"
    );
    0
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
pub(super) fn inet_pton(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    af: i32,
    src: Ptr,
    dst: Ptr,
) -> i32 {
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
pub(super) fn inet_ntop(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    af: i32,
    src: Ptr,
    dst: Ptr,
    size: u32,
) -> u32 {
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
    if st.net_resolver_error(rid).is_some() {
        0
    } else {
        fail(st, SCE_NET_EBADF)
    }
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
pub(super) fn get_mac_address(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    addr: Ptr,
    _flags: i32,
) -> i32 {
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
pub(super) fn resolver_get_error(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    rid: i32,
    result: Ptr,
) -> i32 {
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
/// A REAL BLOCKING WAIT. No socket can ever become readable or writable (nothing connects), so
/// the wait parks the caller for `timeout` MICROSECONDS of guest time and returns ZERO events -
/// the timeout answer, not an error. A negative timeout waits until [`epoll_abort`] or
/// `sceNetEpollDestroy` interrupts it (EINTR); 0 is a poll. A preserved abort interrupts the
/// next wait at once.
///
/// MEASURED: answering at once made a title's network host thread spin ~160 waits a frame - 17 s
/// of a 293 s run busy in a loop that was asking to sleep.
pub(super) fn epoll_wait(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let (eid, events, maxevents, timeout) = (
        ctx.arg(0) as i32,
        ctx.arg(1),
        ctx.arg(2) as i32,
        ctx.arg(3) as i32,
    );
    let r = if !st.net_epoll_exists(eid) {
        fail(st, SCE_NET_EBADF)
    } else if events == 0 || maxevents <= 0 {
        fail(st, SCE_NET_EINVAL)
    } else if st.net_epoll_take_abort(eid) {
        fail(st, SCE_NET_EINTR)
    } else {
        0
    };
    ctx.ret(r as u32);
    if r != 0 || timeout == 0 || !st.is_preemptive() {
        return SvcOutcome::Continue;
    }
    st.net_epoll_park(eid, (timeout > 0).then_some(timeout as u64));
    SvcOutcome::Block
}

/// int sceNetEpollAbort(int eid, int flags)
///
/// Interrupts every thread waiting on `eid`: their waits return EINTR. With no waiter, a
/// PRESERVATION abort is kept for the next wait, which it interrupts; without the flag there is
/// nothing to interrupt, which the library answers with ENOTBLK. The registrations are untouched
/// either way.
#[hostcall]
pub(super) fn epoll_abort(st: &mut VitaState, eid: i32, flags: i32) -> i32 {
    if !st.net_epoll_exists(eid) {
        fail(st, SCE_NET_EBADF)
    } else if flags & !EPOLL_ABORT_FLAG_PRESERVATION != 0 {
        fail(st, SCE_NET_EINVAL)
    } else if st.net_epoll_wake(eid, NET_FAIL as u32) > 0 {
        // The woken waits fail with EINTR.
        st.net_set_errno(SCE_NET_EINTR);
        0
    } else if flags & EPOLL_ABORT_FLAG_PRESERVATION != 0 {
        st.net_epoll_latch_abort(eid);
        0
    } else {
        fail(st, SCE_NET_ENOTBLK)
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
pub(super) fn adhoc_matching_send_data(
    st: &mut VitaState,
    id: i32,
    _addr: Ptr,
    _data_len: i32,
    _data: Ptr,
) -> i32 {
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
pub(super) fn adhoc_matching_set_hello_opt(
    st: &mut VitaState,
    id: i32,
    _opt_len: i32,
    _opt: Ptr,
) -> i32 {
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
pub(super) fn adhoc_matching_select_target(
    st: &mut VitaState,
    id: i32,
    _target: Ptr,
    _opt_len: i32,
    _opt: Ptr,
) -> i32 {
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

// --- SceLibRudp: reliable UDP, over a network that is down -------------------------
//
// RUDP layers numbered, acknowledged streams ("contexts") over a SceNet UDP socket. The same
// line SceNet draws applies one layer up: everything LOCAL is real - the library's
// initialisation and pool, its event handler, contexts created, bound to a socket and
// terminated, poll sets - and the first call that needs a PEER reports that none can be
// reached. No context ever leaves its initial state, so every question about traffic has a
// true answer of zero: nothing readable, nothing writable, no packet counted, no event to
// poll.
//
// The error facility (0x077, `SCE_ERROR_FACILITY_RUDP`) is from the local wiki's error-code
// page. No local source lists the codes inside it; the values here are the RUDP library's
// numbering as published for its PS3 release, which the Vita library is a port of - CHECK any
// of them against a console trace before relying on one beyond "it is negative".

const SCE_RUDP_ERROR_NOT_INITIALIZED: i32 = 0x8077_0001u32 as i32;
const SCE_RUDP_ERROR_ALREADY_INITIALIZED: i32 = 0x8077_0002u32 as i32;
const SCE_RUDP_ERROR_INVALID_CONTEXT_ID: i32 = 0x8077_0003u32 as i32;
const SCE_RUDP_ERROR_INVALID_ARGUMENT: i32 = 0x8077_0004u32 as i32;
const SCE_RUDP_ERROR_INVALID_OPTION: i32 = 0x8077_0005u32 as i32;
const SCE_RUDP_ERROR_CONN_TIMEOUT: i32 = 0x8077_000Bu32 as i32;
const SCE_RUDP_ERROR_NOT_ACCEPTABLE: i32 = 0x8077_0011u32 as i32;
const SCE_RUDP_ERROR_NOT_BOUND: i32 = 0x8077_0013u32 as i32;
const SCE_RUDP_ERROR_WOULDBLOCK: i32 = 0x8077_0016u32 as i32;
const SCE_RUDP_ERROR_ALREADY_BOUND: i32 = 0x8077_001Du32 as i32;
const SCE_RUDP_ERROR_INVALID_POLL_ID: i32 = 0x8077_001Fu32 as i32;

/// One RUDP context, and the socket and virtual port it was bound to. Its event handler is
/// not kept: a context that never opens has no event to call it with.
struct RudpContext {
    id: i32,
    bound: Option<(i32, u16)>,
    /// Options as the guest set them on this context, by option number - see
    /// [`rudp_get_option`].
    options: std::collections::BTreeMap<i32, Vec<u8>>,
}

/// SceLibRudp's state. Ids start at 1 and are never reused within a run.
#[derive(Default)]
pub struct Rudp {
    /// The pool `sceRudpInit` was handed, as `(address, size)`, while initialised.
    pool: Option<(u32, u32)>,
    contexts: Vec<RudpContext>,
    polls: Vec<i32>,
    next_id: i32,
}

impl Rudp {
    fn mint(&mut self) -> i32 {
        self.next_id += 1;
        self.next_id
    }

    fn context(&mut self, id: i32) -> Option<&mut RudpContext> {
        self.contexts.iter_mut().find(|c| c.id == id)
    }
}

/// The library's own answer for a call before `sceRudpInit`: `Err(NOT_INITIALIZED)`.
fn rudp(st: &mut VitaState) -> Result<&mut Rudp, i32> {
    let r = &mut st.net_libs.rudp;
    if r.pool.is_some() {
        Ok(r)
    } else {
        Err(SCE_RUDP_ERROR_NOT_INITIALIZED)
    }
}

/// The context `id` names, or the library's error for why there is none.
fn rudp_context(st: &mut VitaState, id: i32) -> Result<&mut RudpContext, i32> {
    rudp(st)?
        .context(id)
        .ok_or(SCE_RUDP_ERROR_INVALID_CONTEXT_ID)
}

/// Collapse a `Result` carrying the call's return into it.
fn rc(r: Result<i32, i32>) -> i32 {
    r.unwrap_or_else(|e| e)
}

/// int sceRudpInit(void *memPool, int memPoolSize)
///
/// The library runs out of a pool the caller provides; initialising records it. A missing or
/// empty pool is the caller's error, a second init the library's ALREADY_INITIALIZED.
#[hostcall]
pub(super) fn rudp_init(st: &mut VitaState, pool: u32, size: i32) -> i32 {
    let r = &mut st.net_libs.rudp;
    if r.pool.is_some() {
        SCE_RUDP_ERROR_ALREADY_INITIALIZED
    } else if pool == 0 || size <= 0 {
        SCE_RUDP_ERROR_INVALID_ARGUMENT
    } else {
        *r = Rudp {
            pool: Some((pool, size as u32)),
            ..Default::default()
        };
        0
    }
}

/// int sceRudpEnd(void)
///
/// Tears the library down with everything created under it - contexts and poll sets.
#[hostcall]
pub(super) fn rudp_end(st: &mut VitaState) -> i32 {
    rc(rudp(st).map(|r| {
        *r = Rudp::default();
        0
    }))
}

/// int sceRudpEnableInternalIOThread(SceUInt32 stackSize, SceUInt32 priority)
///
/// Asks the library to service its sockets on a thread of its own instead of the title's
/// `sceRudpProcessEvents` calls. Accepted, and no thread is created: that thread's whole job is
/// reading packets off bound sockets and running retransmit timers, and with the interface down
/// no packet arrives and no context ever has anything in flight - so it would only ever sleep,
/// and a guest-visible thread that only sleeps is not something a title can observe the
/// absence of.
#[hostcall]
pub(super) fn rudp_enable_internal_io_thread(st: &mut VitaState, _stack: u32, _prio: u32) -> i32 {
    rc(rudp(st).map(|_| 0))
}

/// int sceRudpSetEventHandler(SceRudpEventHandler handler, void *arg)
///
/// Accepted. The library-wide events it would deliver are about traffic on bound sockets, and
/// there is none, so the handler is never called - and not kept.
#[hostcall]
pub(super) fn rudp_set_event_handler(st: &mut VitaState, _handler: u32, _arg: u32) -> i32 {
    rc(rudp(st).map(|_| 0))
}

/// int sceRudpSetOption(int ctxId, int option, const void *optval, size_t optlen)
///
/// A PER-CONTEXT option: FOUR arguments, the first the id `sceRudpCreateContext` just handed
/// back - read off a title that creates a context and immediately sets option 2 on it to a
/// 4-byte value. (The PS4 library's form takes no context; this one does.) The value is kept
/// exactly as given, to be read back by [`rudp_get_option`]. Nothing here acts on an option:
/// each tunes retransmission, buffering or keep-alive of traffic that never flows.
#[hostcall]
pub(super) fn rudp_set_option(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    id: i32,
    option: i32,
    optval: Ptr,
    optlen: u32,
) -> i32 {
    let bytes = if optval.is_null() {
        None
    } else {
        Some(ctx.read_bytes(optval.addr(), optlen.min(64) as usize))
    };
    rc(rudp_context(st, id).and_then(|c| match bytes {
        None => Err(SCE_RUDP_ERROR_INVALID_ARGUMENT),
        Some(b) => {
            c.options.insert(option, b);
            Ok(0)
        }
    }))
}

/// int sceRudpGetOption(int ctxId, int option, void *optval, size_t optlen)
///
/// [`rudp_set_option`]'s reader, in the same four-argument form. An option the title SET reads
/// back as it set it. One it never set would read back the library's default, and no local
/// source gives any option's default - so that is refused with INVALID_OPTION and said out
/// loud rather than answered with an invented value.
#[hostcall]
pub(super) fn rudp_get_option(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    id: i32,
    option: i32,
    optval: Ptr,
    optlen: u32,
) -> i32 {
    let got = rudp_context(st, id).and_then(|c| {
        if optval.is_null() {
            Err(SCE_RUDP_ERROR_INVALID_ARGUMENT)
        } else {
            c.options
                .get(&option)
                .cloned()
                .ok_or(SCE_RUDP_ERROR_INVALID_OPTION)
        }
    });
    match got {
        Ok(b) => {
            let n = b.len().min(optlen as usize);
            ctx.write_bytes(optval.addr(), &b[..n]);
            0
        }
        Err(e) => {
            if e == SCE_RUDP_ERROR_INVALID_OPTION {
                tracing::warn!(
                    target: "vitaslop::net",
                    option,
                    "sceRudpGetOption: the title asked for an option it never set - its default \
                     is not established, so the call is refused rather than answered"
                );
            }
            e
        }
    }
}

/// int sceRudpCreateContext(SceRudpContextEventHandler handler, void *arg, int *ctxId)
///
/// A context is local until it is bound and initiated, so it is really created.
#[hostcall]
pub(super) fn rudp_create_context(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    _handler: u32,
    _arg: u32,
    out: Ptr,
) -> i32 {
    let made = rudp(st).and_then(|r| {
        if out.is_null() {
            Err(SCE_RUDP_ERROR_INVALID_ARGUMENT)
        } else {
            let id = r.mint();
            r.contexts.push(RudpContext {
                id,
                bound: None,
                options: Default::default(),
            });
            Ok(id)
        }
    });
    match made {
        Ok(id) => {
            ctx.write_u32(out.addr(), id as u32);
            0
        }
        Err(e) => e,
    }
}

/// int sceRudpBind(int ctxId, int soc, SceUInt16 vport, SceUInt8 muxMode)
///
/// Associates the context with one of the title's SceNet sockets and a virtual port on it:
/// local, so recorded. A context binds once.
#[hostcall]
pub(super) fn rudp_bind(st: &mut VitaState, id: i32, soc: i32, vport: u32, _mux: u32) -> i32 {
    rc(rudp_context(st, id).and_then(|c| match c.bound {
        Some(_) => Err(SCE_RUDP_ERROR_ALREADY_BOUND),
        None => {
            c.bound = Some((soc, vport as u16));
            Ok(0)
        }
    }))
}

/// int sceRudpInitiate(int ctxId, const SceNetSockaddr *to, size_t toLen, SceUInt16 vport)
/// int sceRudpActivate(int ctxId, const SceNetSockaddr *to, size_t toLen)
///
/// The two ways a bound context opens a connection - the active and the passive side - and
/// the first calls that need a PEER. With the interface down the handshake cannot reach one
/// (the socket under the context cannot send - see the module docs), and a handshake that gets
/// no answer ends in CONN_TIMEOUT. It is reported here, at the call, because this engine runs
/// no RUDP retransmit timer for it to expire on later; the context stays as it was, unopened.
/// An unbound context is the library's NOT_BOUND first.
#[hostcall]
pub(super) fn rudp_connect(st: &mut VitaState, id: i32) -> i32 {
    rc(rudp_context(st, id).and_then(|c| match c.bound {
        None => Err(SCE_RUDP_ERROR_NOT_BOUND),
        Some(_) => Err(SCE_RUDP_ERROR_CONN_TIMEOUT),
    }))
}

/// int sceRudpTerminate(int ctxId)
///
/// Releases the context. Nothing was ever open on it, so there is nothing to close first.
#[hostcall]
pub(super) fn rudp_terminate(st: &mut VitaState, id: i32) -> i32 {
    rc(rudp(st).and_then(|r| {
        let before = r.contexts.len();
        r.contexts.retain(|c| c.id != id);
        if r.contexts.len() == before {
            Err(SCE_RUDP_ERROR_INVALID_CONTEXT_ID)
        } else {
            Ok(0)
        }
    }))
}

/// Zero `size` bytes at `out`: a status struct every field of which is truly zero here.
fn zero_status(ctx: &mut GuestCtx, out: Ptr, size: u32) {
    ctx.write_bytes(out.addr(), &vec![0u8; size.min(0x1000) as usize]);
}

/// int sceRudpGetContextStatus(int ctxId, SceRudpContextStatus *status, size_t statusSize)
///
/// No local source gives this struct's layout, and none is needed: a context that was never
/// opened is in the INITIAL state (the zero of the library's state enumeration) and every
/// counter it keeps - bytes and packets sent, received, retransmitted - is zero. So the
/// caller's struct, at the size the caller declares, is zeroed and that is the whole answer.
#[hostcall]
pub(super) fn rudp_get_context_status(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    id: i32,
    out: Ptr,
    size: u32,
) -> i32 {
    let r = rudp_context(st, id).and_then(|_| {
        if out.is_null() {
            Err(SCE_RUDP_ERROR_INVALID_ARGUMENT)
        } else {
            Ok(0)
        }
    });
    if r == Ok(0) {
        zero_status(ctx, out, size);
    }
    rc(r)
}

/// int sceRudpGetStatus(SceRudpStatus *status, size_t statusSize)
///
/// The library-wide traffic counters, for the same reason as [`rudp_get_context_status`]: all
/// zero, at the caller's declared size.
#[hostcall]
pub(super) fn rudp_get_status(ctx: &mut GuestCtx, st: &mut VitaState, out: Ptr, size: u32) -> i32 {
    let r = rudp(st).and_then(|_| {
        if out.is_null() {
            Err(SCE_RUDP_ERROR_INVALID_ARGUMENT)
        } else {
            Ok(0)
        }
    });
    if r == Ok(0) {
        zero_status(ctx, out, size);
    }
    rc(r)
}

/// int sceRudpGetSizeReadable(int ctxId) / sceRudpGetSizeWritable(int ctxId)
///
/// Bytes that can be read now / written now: none of either, on a context that never opened.
#[hostcall]
pub(super) fn rudp_get_size(st: &mut VitaState, id: i32) -> i32 {
    rc(rudp_context(st, id).map(|_| 0))
}

/// int sceRudpRead(int ctxId, void *data, size_t len, SceUInt8 flags, SceRudpReadInfo *info)
///
/// Nothing has arrived and nothing will: the non-blocking answer, WOULDBLOCK.
#[hostcall]
pub(super) fn rudp_read(st: &mut VitaState, id: i32) -> i32 {
    rc(rudp_context(st, id).and(Err(SCE_RUDP_ERROR_WOULDBLOCK)))
}

/// int sceRudpWrite(int ctxId, const void *data, size_t len, SceUInt8 flags)
///
/// A context only accepts data once it is open, and this one never opens: NOT_ACCEPTABLE.
#[hostcall]
pub(super) fn rudp_write(st: &mut VitaState, id: i32) -> i32 {
    rc(rudp_context(st, id).and(Err(SCE_RUDP_ERROR_NOT_ACCEPTABLE)))
}

/// int sceRudpFlush(int ctxId)
///
/// Nothing was ever queued on the context, so there is nothing to push out: done.
#[hostcall]
pub(super) fn rudp_flush(st: &mut VitaState, id: i32) -> i32 {
    rc(rudp_context(st, id).map(|_| 0))
}

/// int sceRudpProcessEvents(SceUInt32 timeout)
///
/// The title's own pump, for when it services the sockets itself: there are no packets and no
/// timers, so there are no events to process.
#[hostcall]
pub(super) fn rudp_process_events(st: &mut VitaState, _timeout: u32) -> i32 {
    rc(rudp(st).map(|_| 0))
}

/// int sceRudpNetReceived(int soc, const SceUInt8 *data, size_t dataLen,
///                        const SceNetSockaddr *from, size_t fromLen)
///
/// The title hands RUDP a datagram it received itself (multiplexed mode). None can arrive on
/// a down interface; were one handed over, it would name no context this library opened, and
/// RUDP drops such a datagram as UDP would. The call validates the buffer and reports success.
#[hostcall]
pub(super) fn rudp_net_received(st: &mut VitaState, _soc: i32, data: u32, len: u32) -> i32 {
    let valid = if data == 0 || len == 0 {
        Err(SCE_RUDP_ERROR_INVALID_ARGUMENT)
    } else {
        Ok(0)
    };
    rc(rudp(st).and(valid))
}

/// int sceRudpPollCreate(size_t size)
///
/// A poll set is local: really created, and its id returned.
#[hostcall]
pub(super) fn rudp_poll_create(st: &mut VitaState, _size: u32) -> i32 {
    rc(rudp(st).map(|r| {
        let id = r.mint();
        r.polls.push(id);
        id
    }))
}

/// int sceRudpPollDestroy(int pollId)
#[hostcall]
pub(super) fn rudp_poll_destroy(st: &mut VitaState, id: i32) -> i32 {
    rc(rudp(st).and_then(|r| {
        let before = r.polls.len();
        r.polls.retain(|&p| p != id);
        if r.polls.len() == before {
            Err(SCE_RUDP_ERROR_INVALID_POLL_ID)
        } else {
            Ok(0)
        }
    }))
}

/// int sceRudpPollControl(int pollId, int op, int ctxId, SceUInt16 events)
///
/// Adds, modifies or removes a context's interest in the set. Both ids are checked; the
/// interest itself is not kept, because no context can ever raise an event for it to select.
#[hostcall]
pub(super) fn rudp_poll_control(st: &mut VitaState, poll: i32, _op: i32, id: i32) -> i32 {
    rc(rudp(st).and_then(|r| {
        if !r.polls.contains(&poll) {
            Err(SCE_RUDP_ERROR_INVALID_POLL_ID)
        } else if r.context(id).is_none() {
            Err(SCE_RUDP_ERROR_INVALID_CONTEXT_ID)
        } else {
            Ok(0)
        }
    }))
}

/// int sceRudpPollWait(int pollId, SceRudpPollEvent *events, size_t eventLen,
///                     SceRudpUsec timeout)
///
/// A REAL WAIT, as `sceNetEpollWait` is (see [`epoll_wait`] for what answering at once did to a
/// polling thread): no context can become readable or writable, so the caller is parked for
/// `timeout` microseconds of guest time and gets zero events - the timeout answer. A zero
/// timeout is a poll and returns at once.
pub(super) fn rudp_poll_wait(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let (poll, timeout) = (ctx.arg(0) as i32, ctx.arg(3));
    let r = rc(rudp(st).and_then(|r| {
        if r.polls.contains(&poll) {
            Ok(0)
        } else {
            Err(SCE_RUDP_ERROR_INVALID_POLL_ID)
        }
    }));
    ctx.ret(r as u32);
    if r != 0 || timeout == 0 || !st.is_preemptive() {
        return SvcOutcome::Continue;
    }
    st.sleep_park(u64::from(timeout));
    SvcOutcome::Block
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::{SliceMemory, VFP_ARG_COUNT};
    use crate::world::DeterministicWorld;
    use vitaslop_transpiler::abi::REG_COUNT;

    type Entry = fn(&mut GuestCtx, &mut VitaState);

    /// [`epoll_wait`] as an [`Entry`] (it answers a scheduler outcome the harness ignores).
    fn wait(ctx: &mut GuestCtx, st: &mut VitaState) {
        let _ = epoll_wait(ctx, st);
    }

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

        fn call(&mut self, f: Entry, args: &[u32]) -> i32 {
            self.regs = [0; REG_COUNT];
            self.regs[..args.len()].copy_from_slice(args);
            {
                let mut mem = SliceMemory(&mut self.bytes);
                let mut ctx = GuestCtx::new(&mut self.regs, &mut self.vfp, &mut mem, 0);
                f(&mut ctx, &mut self.st);
            }
            self.regs[0] as i32
        }
    }

    /// A preserved abort interrupts the next wait once and only once; an abort with nothing to
    /// interrupt and no PRESERVATION is refused; a bad id or an unknown flag is refused.
    #[test]
    fn a_preserved_abort_interrupts_the_next_wait_once() {
        let mut r = Run::new();
        let eid = r.call(epoll_create, &[0, 0]) as u32;
        assert_eq!(r.call(wait, &[eid, 0x100, 1, 0]), 0);
        assert_eq!(r.call(epoll_abort, &[eid, 1]), 0);
        assert_eq!(r.call(wait, &[eid, 0x100, 1, 0]), NET_FAIL);
        assert_eq!(r.call(wait, &[eid, 0x100, 1, 0]), 0);
        assert_eq!(r.call(epoll_abort, &[eid, 0]), NET_FAIL);
        assert_eq!(r.call(wait, &[eid, 0x100, 1, 0]), 0);
        assert_eq!(r.call(epoll_abort, &[eid, 2]), NET_FAIL);
        assert_eq!(r.call(epoll_abort, &[0x9999, 1]), NET_FAIL);
    }

    /// A parked wait is interrupted by an abort with EINTR (no latch is left behind); one whose
    /// timeout runs out is no longer a waiter, so a later abort finds nobody to interrupt.
    #[test]
    fn an_abort_interrupts_a_parked_wait_and_a_timed_out_wait_is_gone() {
        let mut r = Run::new();
        let eid = r.call(epoll_create, &[0, 0]);
        let me = r.st.current_thread();
        r.st.net_epoll_park(eid, None);
        assert_eq!(
            r.call(epoll_abort, &[eid as u32, 0]),
            0,
            "a waiter was interrupted"
        );
        assert_eq!(r.st.take_resume_code(me), Some(NET_FAIL as u32));
        assert_eq!(
            r.call(wait, &[eid as u32, 0x100, 1, 0]),
            0,
            "no latch was left"
        );
        r.st.net_epoll_park(eid, Some(1_000));
        r.st.advance_time_to(5_000);
        assert_eq!(
            r.st.take_resume_code(me),
            None,
            "a wait that ran out returns its 0 events"
        );
        assert_eq!(
            r.call(epoll_abort, &[eid as u32, 0]),
            NET_FAIL,
            "nobody is waiting any more"
        );
        r.st.net_epoll_park(eid, Some(1_000_000));
        assert_eq!(r.call(epoll_destroy, &[eid as u32]), 0);
        assert_eq!(
            r.st.take_resume_code(me),
            Some(NET_FAIL as u32),
            "a destroy interrupts its waiters"
        );
        // A null event buffer or a non-positive capacity is refused.
        let eid = r.call(epoll_create, &[0, 0]) as u32;
        assert_eq!(r.call(wait, &[eid, 0, 1, 0]), NET_FAIL);
        assert_eq!(r.call(wait, &[eid, 0x100, 0, 0]), NET_FAIL);
    }

    /// Destroying an epoll forgets its latch, and a later epoll never reuses a live id.
    #[test]
    fn epoll_ids_are_never_reused() {
        let mut r = Run::new();
        let a = r.call(epoll_create, &[0, 0]);
        let b = r.call(epoll_create, &[0, 0]);
        assert_eq!(r.call(epoll_abort, &[a as u32, 1]), 0);
        assert_eq!(r.call(epoll_destroy, &[a as u32]), 0);
        let c = r.call(epoll_create, &[0, 0]);
        assert!(c != a && c != b);
        assert_eq!(r.call(wait, &[c as u32, 0x100, 1, 0]), 0);
        assert_eq!(r.call(wait, &[a as u32, 0x100, 1, 0]), NET_FAIL);
    }

    fn word(r: &Run, at: usize) -> u32 {
        u32::from_le_bytes(r.bytes[at..at + 4].try_into().unwrap())
    }

    /// The statistics need an initialised libnet and report its pool with no packet counted;
    /// a term forgets the pool again.
    #[test]
    fn statistics_report_the_init_pool_and_no_traffic() {
        let mut r = Run::new();
        assert_eq!(
            r.call(get_statistics_info, &[0x100, 0]),
            SCE_NET_ERROR_ENOTINIT
        );
        assert_eq!(r.call(net_init, &[0]), SCE_NET_ERROR_EINVAL);
        r.bytes[0x84..0x88].copy_from_slice(&0x8000u32.to_le_bytes()); // param.size
        assert_eq!(r.call(net_init, &[0x80]), 0);
        assert_eq!(r.call(net_init, &[0x80]), SCE_NET_ERROR_EBUSY, "a repeat init");
        r.bytes[0x100..0x118].fill(0xAA);
        assert_eq!(r.call(get_statistics_info, &[0x100, 0]), 0);
        assert_eq!(word(&r, 0x100), NET_KERNEL_HEAP_BYTES, "kernel_mem_free_size");
        assert_eq!(word(&r, 0x104), NET_KERNEL_HEAP_BYTES, "kernel_mem_free_min");
        assert_eq!(word(&r, 0x108), 0, "packet_count");
        assert_eq!(word(&r, 0x10C), 0, "packet_qos_count");
        assert_eq!(word(&r, 0x110), 0x8000, "libnet_mem_free_size");
        assert_eq!(word(&r, 0x114), 0x8000, "libnet_mem_free_min");
        assert_eq!(r.call(get_statistics_info, &[0x100, 1]), 0);
        assert_eq!(word(&r, 0x100), NET_KERNEL_HEAP_BYTES.swap_bytes(), "flags swap the kernel words");
        assert_eq!(word(&r, 0x110), 0x8000, "but not libnet's");
        assert_eq!(r.call(get_statistics_info, &[0, 0]), SCE_NET_ERROR_EINVAL);
        assert_eq!(r.call(net_term, &[]), 0);
        assert_eq!(
            r.call(get_statistics_info, &[0x100, 0]),
            SCE_NET_ERROR_ENOTINIT
        );
    }

    /// RUDP's local objects are real - init once, contexts and poll sets minted and released,
    /// a context binding once - and every call that needs a peer or traffic says there is none.
    #[test]
    fn rudp_local_objects_are_real_and_peers_are_unreachable() {
        let mut r = Run::new();
        let ctx_out = 0x100u32;
        assert_eq!(
            r.call(rudp_create_context, &[0, 0, ctx_out]),
            SCE_RUDP_ERROR_NOT_INITIALIZED
        );
        assert_eq!(r.call(rudp_init, &[0, 0x1000]), SCE_RUDP_ERROR_INVALID_ARGUMENT);
        assert_eq!(r.call(rudp_init, &[0x200, 0x1000]), 0);
        assert_eq!(
            r.call(rudp_init, &[0x200, 0x1000]),
            SCE_RUDP_ERROR_ALREADY_INITIALIZED
        );
        assert_eq!(r.call(rudp_create_context, &[0x8100_0001, 0, ctx_out]), 0);
        let id = word(&r, ctx_out as usize);
        assert!(id > 0);
        assert_eq!(r.call(rudp_connect, &[id]), SCE_RUDP_ERROR_NOT_BOUND);
        assert_eq!(r.call(rudp_bind, &[id, 3, 1, 0]), 0);
        assert_eq!(r.call(rudp_bind, &[id, 3, 1, 0]), SCE_RUDP_ERROR_ALREADY_BOUND);
        assert_eq!(r.call(rudp_connect, &[id]), SCE_RUDP_ERROR_CONN_TIMEOUT);
        assert_eq!(r.call(rudp_get_size, &[id]), 0);
        assert_eq!(r.call(rudp_read, &[id]), SCE_RUDP_ERROR_WOULDBLOCK);
        assert_eq!(r.call(rudp_write, &[id]), SCE_RUDP_ERROR_NOT_ACCEPTABLE);
        assert_eq!(r.call(rudp_flush, &[id]), 0);

        // Status structs are zeroed at the size the caller declares, and no further.
        r.bytes[0x300..0x320].fill(0xAA);
        assert_eq!(r.call(rudp_get_context_status, &[id, 0x300, 0x10]), 0);
        assert_eq!(&r.bytes[0x300..0x310], &[0u8; 0x10]);
        assert_eq!(r.bytes[0x310], 0xAA);

        // Options read back as set; an unset one has no established default.
        r.bytes[0x340..0x344].copy_from_slice(&5u32.to_le_bytes());
        assert_eq!(r.call(rudp_set_option, &[id, 2, 0x340, 4]), 0);
        assert_eq!(r.call(rudp_get_option, &[id, 2, 0x348, 4]), 0);
        assert_eq!(word(&r, 0x348), 5);
        assert_eq!(
            r.call(rudp_get_option, &[id, 9, 0x348, 4]),
            SCE_RUDP_ERROR_INVALID_OPTION
        );
        assert_eq!(
            r.call(rudp_set_option, &[0x77, 2, 0x340, 4]),
            SCE_RUDP_ERROR_INVALID_CONTEXT_ID
        );

        let poll = r.call(rudp_poll_create, &[4]);
        assert!(poll > 0 && poll != id as i32);
        assert_eq!(r.call(rudp_poll_control, &[poll as u32, 1, id, 1]), 0);
        assert_eq!(
            r.call(rudp_poll_control, &[poll as u32, 1, 0x77, 1]),
            SCE_RUDP_ERROR_INVALID_CONTEXT_ID
        );
        assert_eq!(r.call(rudp_poll_destroy, &[poll as u32]), 0);
        assert_eq!(
            r.call(rudp_poll_destroy, &[poll as u32]),
            SCE_RUDP_ERROR_INVALID_POLL_ID
        );

        assert_eq!(r.call(rudp_terminate, &[id]), 0);
        assert_eq!(r.call(rudp_terminate, &[id]), SCE_RUDP_ERROR_INVALID_CONTEXT_ID);
        assert_eq!(r.call(rudp_end, &[]), 0);
        assert_eq!(r.call(rudp_end, &[]), SCE_RUDP_ERROR_NOT_INITIALIZED);
    }
}
