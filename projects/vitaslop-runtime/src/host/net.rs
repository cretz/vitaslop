//! The offline network: SceNet's socket table, resolvers and epolls, SceHttp's object graph,
//! and SceNetAdhocMatching's contexts.
//!
//! [`NetState`] OWNS that state: its fields are private to this file. Everything recorded here
//! is LOCAL state the guest itself supplied - no host socket is ever opened and no request is
//! ever sent - so every id is a pure function of the guest's own call sequence and two runs of
//! one input agree. See `vita::net` and `vita::http` for what is modelled and why. The
//! `VitaState` methods the NID handlers call are thin forwarders in `state_net.rs`; the parts
//! that need the scheduler (parking an epoll waiter, waking it, an errno slot's guest address)
//! are composed there. It needs nothing from the rest of the host.

/// One socket in the offline SceNet table. Everything recorded here is LOCAL state -
/// what the guest itself set - because nothing else exists to record.
struct NetSocket {
    id: i32,
    /// The debug name SceNet takes at creation (it names sockets, unlike BSD).
    name: String,
    domain: i32,
    ty: i32,
    protocol: i32,
    /// The address `sceNetBind` recorded, as `(network-order ip, host-order port)`.
    local: (u32, u16),
    listening: bool,
    /// `(level, optname, value)` for every option the guest set, so a Get round-trips.
    options: Vec<(i32, i32, u32)>,
    closed: bool,
}

/// Socket, resolver and epoll id ranges. Deliberately disjoint from each other and well
/// above the file-descriptor range, so an id used with the wrong API is rejected rather
/// than silently naming somebody else's object.
const NET_FD_BASE: i32 = 0x1000;
const NET_RESOLVER_BASE: i32 = 0x2000;
const NET_EPOLL_BASE: i32 = 0x3000;

/// Which SceHttp object an id names. The three form a strict hierarchy - a connection
/// belongs to a template, a request to a connection - and the kind is what lets a call
/// refuse an id from the wrong level instead of acting on it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HttpKind {
    Template,
    Connection,
    Request,
}

/// One object in the offline SceHttp graph. Like [`NetSocket`], everything recorded is
/// LOCAL state the guest itself supplied; nothing here was learned from a server,
/// because no request is ever sent. See `vita::http`.
struct HttpObject {
    id: i32,
    kind: HttpKind,
    /// The template a connection hangs off, or the connection a request hangs off; 0
    /// for a template. Kept so a delete can be reasoned about and a trace can show the
    /// graph the title built.
    parent: i32,
    /// A template's user agent, or a connection's/request's URL - whatever the guest
    /// named the object with.
    label: String,
    /// The `SceHttpMethods` value a request was created with; 0 for the other kinds.
    method: i32,
    /// Every header the guest added, in order, as `(name, value)`.
    headers: Vec<(String, String)>,
    /// Connect/send/receive timeouts in microseconds, as the guest set them. The
    /// library's own defaults until it does.
    /// Connect, send and receive timeouts, then the resolver's timeout and retry count -
    /// each recorded against the object it was set on.
    timeouts: [u32; 5],
    /// The on/off settings the guest can set and read back, indexed by [`HttpSwitch`].
    switches: [bool; 3],
    deleted: bool,
}

/// A per-object on/off setting of SceHttp: set by `sceHttpSet*`, read back by the matching
/// `sceHttpGet*`. Each is LOCAL state the guest chose, recorded and returned as given.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HttpSwitch {
    /// `sceHttpSetCookieEnabled`. Off until the guest turns it on - the library's default.
    Cookies = 0,
    /// `sceHttpSetAuthEnabled`: answer a 401/407 through the auth callback. On by default.
    Auth = 1,
    /// `sceHttpSetAutoRedirect`: follow a 3xx without asking. On by default.
    AutoRedirect = 2,
}

/// SceHttp id ranges, one per kind and disjoint from each other and from the SceNet
/// ranges above - so an id used at the wrong level is REFUSED rather than silently
/// naming another object. Ids start at 1 within a range because `SCE_HTTP_INVALID_ID`
/// is 0 and a valid id must never collide with it.
const HTTP_TEMPLATE_BASE: i32 = 0x4000;
const HTTP_CONNECTION_BASE: i32 = 0x5000;
const HTTP_REQUEST_BASE: i32 = 0x6000;

/// The offline network's whole state. See the module docs.
#[derive(Default)]
pub(super) struct NetState {
    /// The offline SceNet socket table, its resolver and epoll handles, and the
    /// per-thread errno slots. See `vita::net` for the model.
    sockets: Vec<NetSocket>,
    /// `sceNetAdhocMatchingInit`: the pool the guest handed over, once. `None` until it
    /// does, which is what makes a later `Create` reportable as NOT_INITIALIZED rather
    /// than quietly working.
    adhoc_matching_pool: Option<(u32, u32)>,
    /// Live `sceNetAdhocMatchingCreate` contexts, as `(id, started)`. Kept so an id the
    /// guest never got cannot be stopped or deleted, and so the ids are a function of the
    /// call sequence alone (deterministic across runs).
    adhoc_matchings: Vec<(i32, bool)>,
    resolvers: Vec<(i32, i32)>,
    epolls: Vec<(i32, Vec<i32>)>,
    /// Epolls carrying a preserved `sceNetEpollAbort`, which the next wait on them consumes.
    epoll_aborted: Vec<i32>,
    /// Threads parked in `sceNetEpollWait`, as `(thid, eid)`. A timed wait is ALSO in
    /// `sleep_waiters` (its deadline wakes it with 0 events); an untimed one waits only here.
    /// `sceNetEpollAbort` / `Destroy` wake every waiter on the epoll with EINTR.
    epoll_waiters: Vec<(i32, i32)>,
    /// Epolls created so far, destroyed ones included, so an id is never handed out twice.
    epoll_minted: i32,
    errno: Vec<(i32, u32)>,
    /// The offline SceHttp object graph: templates, connections and requests. See
    /// `vita::http` for what is real in it and what reports the link being down.
    http_objects: Vec<HttpObject>,
    /// The `SCE_HTTPS_FLAG_*` bits the guest turned ON and OFF (`sceHttpsEnableOption`,
    /// `sceHttpsDisableOption`). Kept as the two sets it named rather than one word over
    /// an assumed default: no handshake ever happens to consume them, and a trace should
    /// show what the title asked for, not our guess at what it started from.
    https_options: (u32, u32),
}

impl NetState {
    // --- SceNetAdhocMatching: the peer table that is always empty ---

    pub(super) fn adhoc_matching_init(&mut self, pool_ptr: u32, pool_size: u32) -> bool {
        if self.adhoc_matching_pool.is_some() {
            return false;
        }
        self.adhoc_matching_pool = Some((pool_ptr, pool_size));
        true
    }

    pub(super) fn adhoc_matching_ready(&self) -> bool {
        self.adhoc_matching_pool.is_some()
    }

    pub(super) fn adhoc_matching_term(&mut self) -> bool {
        if self.adhoc_matching_pool.take().is_none() {
            return false;
        }
        self.adhoc_matchings.clear();
        true
    }

    pub(super) fn adhoc_matching_create(&mut self) -> i32 {
        let id = self.adhoc_matchings.len() as i32 + 1;
        self.adhoc_matchings.push((id, false));
        id
    }

    pub(super) fn adhoc_matching_set_started(&mut self, id: i32, started: bool) -> Option<()> {
        let slot = self.adhoc_matchings.iter_mut().find(|(i, _)| *i == id)?;
        slot.1 = started;
        Some(())
    }

    pub(super) fn adhoc_matching_live(&self, id: i32) -> bool {
        self.adhoc_matchings.iter().any(|(i, _)| *i == id)
    }

    pub(super) fn adhoc_matching_delete(&mut self, id: i32) -> bool {
        let before = self.adhoc_matchings.len();
        self.adhoc_matchings.retain(|(i, _)| *i != id);
        self.adhoc_matchings.len() != before
    }

    // --- SceNet: the offline socket table ---

    pub(super) fn socket(&mut self, name: &str, domain: i32, ty: i32, protocol: i32) -> i32 {
        let id = NET_FD_BASE + self.sockets.len() as i32;
        self.sockets.push(NetSocket {
            id,
            name: name.to_string(),
            domain,
            ty,
            protocol,
            local: (0, 0),
            listening: false,
            options: Vec::new(),
            closed: false,
        });
        id
    }

    fn socket_mut(&mut self, id: i32) -> Option<&mut NetSocket> {
        self.sockets.iter_mut().find(|s| s.id == id && !s.closed)
    }

    pub(super) fn socket_exists(&self, id: i32) -> bool {
        self.sockets.iter().any(|s| s.id == id && !s.closed)
    }

    pub(super) fn close(&mut self, id: i32) -> bool {
        match self.socket_mut(id) {
            Some(s) => {
                s.closed = true;
                true
            }
            None => false,
        }
    }

    pub(super) fn bind(&mut self, id: i32, ip: u32, port: u16) -> bool {
        match self.socket_mut(id) {
            Some(s) => {
                s.local = (ip, port);
                true
            }
            None => false,
        }
    }

    pub(super) fn listen(&mut self, id: i32) -> bool {
        match self.socket_mut(id) {
            Some(s) => {
                s.listening = true;
                true
            }
            None => false,
        }
    }

    pub(super) fn local_addr(&self, id: i32) -> Option<(u32, u16)> {
        self.sockets
            .iter()
            .find(|s| s.id == id && !s.closed)
            .map(|s| s.local)
    }

    pub(super) fn set_opt(&mut self, id: i32, level: i32, name: i32, value: u32) -> bool {
        match self.socket_mut(id) {
            Some(s) => {
                s.options.retain(|&(l, n, _)| (l, n) != (level, name));
                s.options.push((level, name, value));
                true
            }
            None => false,
        }
    }

    pub(super) fn get_opt(&self, id: i32, level: i32, name: i32) -> Option<u32> {
        let s = self.sockets.iter().find(|s| s.id == id && !s.closed)?;
        Some(
            s.options
                .iter()
                .find(|&&(l, n, _)| (l, n) == (level, name))
                .map(|&(_, _, v)| v)
                .unwrap_or(0),
        )
    }

    pub(super) fn netstat(&self) -> String {
        use std::fmt::Write;
        let mut s = String::from("netstat: offline model, no host sockets\n");
        for k in self.sockets.iter().filter(|s| !s.closed) {
            let (ip, port) = k.local;
            let b = ip.to_le_bytes();
            let _ = writeln!(
                s,
                "  fd={} name={:?} domain={} type={} proto={} local={}.{}.{}.{}:{}{}",
                k.id,
                k.name,
                k.domain,
                k.ty,
                k.protocol,
                b[0],
                b[1],
                b[2],
                b[3],
                port,
                if k.listening { " LISTEN" } else { "" }
            );
        }
        s
    }

    /// The `sceNetErrnoLoc` slot already allocated for thread `thid`, if any.
    pub(super) fn errno_slot(&self, thid: i32) -> Option<u32> {
        self.errno
            .iter()
            .find(|&&(t, _)| t == thid)
            .map(|&(_, addr)| addr)
    }

    /// Record the `sceNetErrnoLoc` slot just allocated for thread `thid`.
    pub(super) fn set_errno_slot(&mut self, thid: i32, addr: u32) {
        self.errno.push((thid, addr));
    }

    // --- SceHttp: the offline object graph ---

    pub(super) fn http_create(
        &mut self,
        kind: HttpKind,
        parent: i32,
        label: &str,
        method: i32,
    ) -> i32 {
        let base = match kind {
            HttpKind::Template => HTTP_TEMPLATE_BASE,
            HttpKind::Connection => HTTP_CONNECTION_BASE,
            HttpKind::Request => HTTP_REQUEST_BASE,
        };
        // Count objects of THIS kind, deleted ones included, so an id is never reused
        // within a run: a title that keeps a stale id must see it refused, not find it
        // pointing at somebody else's object.
        let n = self.http_objects.iter().filter(|o| o.kind == kind).count() as i32;
        let id = base + 1 + n;
        self.http_objects.push(HttpObject {
            id,
            kind,
            parent,
            label: label.to_string(),
            method,
            headers: Vec::new(),
            timeouts: [0; 5],
            switches: [false, true, true],
            deleted: false,
        });
        id
    }

    pub(super) fn http_set_switch(&mut self, id: i32, which: HttpSwitch, on: bool) -> bool {
        match self.http_object_mut(id) {
            Some(o) => {
                o.switches[which as usize] = on;
                true
            }
            None => false,
        }
    }

    pub(super) fn http_switch(&self, id: i32, which: HttpSwitch) -> Option<bool> {
        self.http_objects
            .iter()
            .find(|o| o.id == id && !o.deleted)
            .map(|o| o.switches[which as usize])
    }

    pub(super) fn https_set_options(&mut self, flags: u32, on: bool) {
        let (set, clear) = &mut self.https_options;
        if on {
            *set |= flags;
            *clear &= !flags;
        } else {
            *clear |= flags;
            *set &= !flags;
        }
    }

    pub(super) fn https_options(&self) -> (u32, u32) {
        self.https_options
    }

    fn http_object_mut(&mut self, id: i32) -> Option<&mut HttpObject> {
        self.http_objects
            .iter_mut()
            .find(|o| o.id == id && !o.deleted)
    }

    pub(super) fn http_exists(&self, id: i32) -> bool {
        self.http_objects.iter().any(|o| o.id == id && !o.deleted)
    }

    pub(super) fn http_is(&self, id: i32, kind: HttpKind) -> bool {
        self.http_objects
            .iter()
            .any(|o| o.id == id && o.kind == kind && !o.deleted)
    }

    pub(super) fn http_delete(&mut self, id: i32) -> bool {
        let Some(o) = self.http_object_mut(id) else {
            return false;
        };
        o.deleted = true;
        let mut doomed = vec![id];
        while let Some(parent) = doomed.pop() {
            let kids: Vec<i32> = self
                .http_objects
                .iter()
                .filter(|o| o.parent == parent && !o.deleted)
                .map(|o| o.id)
                .collect();
            for k in kids {
                if let Some(o) = self.http_object_mut(k) {
                    o.deleted = true;
                }
                doomed.push(k);
            }
        }
        true
    }

    pub(super) fn http_set_timeout(&mut self, id: i32, which: usize, usec: u32) -> bool {
        match self.http_object_mut(id) {
            Some(o) if which < o.timeouts.len() => {
                o.timeouts[which] = usec;
                true
            }
            _ => false,
        }
    }

    pub(super) fn http_add_header(&mut self, id: i32, name: &str, value: &str) -> bool {
        match self.http_object_mut(id) {
            Some(o) => {
                o.headers.push((name.to_string(), value.to_string()));
                true
            }
            None => false,
        }
    }

    pub(super) fn http_report(&self) -> Vec<String> {
        self.http_objects
            .iter()
            .filter(|o| !o.deleted)
            .map(|o| {
                format!(
                    "{:?} {:#x} parent={:#x} method={} timeouts={:?} headers={} label={:?}",
                    o.kind,
                    o.id,
                    o.parent,
                    o.method,
                    o.timeouts,
                    o.headers.len(),
                    o.label
                )
            })
            .collect()
    }

    // --- resolvers ---

    pub(super) fn resolver_create(&mut self) -> i32 {
        let id = NET_RESOLVER_BASE + self.resolvers.len() as i32;
        self.resolvers.push((id, 0));
        id
    }

    pub(super) fn resolver_destroy(&mut self, rid: i32) -> bool {
        let before = self.resolvers.len();
        self.resolvers.retain(|&(id, _)| id != rid);
        self.resolvers.len() != before
    }

    pub(super) fn resolver_set_error(&mut self, rid: i32, code: i32) {
        if let Some(e) = self.resolvers.iter_mut().find(|(id, _)| *id == rid) {
            e.1 = code;
        }
    }

    pub(super) fn resolver_error(&self, rid: i32) -> Option<i32> {
        self.resolvers
            .iter()
            .find(|(id, _)| *id == rid)
            .map(|&(_, e)| e)
    }

    // --- epolls ---

    pub(super) fn epoll_create(&mut self) -> i32 {
        let id = NET_EPOLL_BASE + self.epoll_minted;
        self.epoll_minted += 1;
        self.epolls.push((id, Vec::new()));
        id
    }

    pub(super) fn epoll_latch_abort(&mut self, eid: i32) -> bool {
        if !self.epoll_exists(eid) {
            return false;
        }
        if !self.epoll_aborted.contains(&eid) {
            self.epoll_aborted.push(eid);
        }
        true
    }

    pub(super) fn epoll_take_abort(&mut self, eid: i32) -> bool {
        let before = self.epoll_aborted.len();
        self.epoll_aborted.retain(|&e| e != eid);
        self.epoll_aborted.len() != before
    }

    pub(super) fn epoll_exists(&self, eid: i32) -> bool {
        self.epolls.iter().any(|(id, _)| *id == eid)
    }

    /// Forget epoll `eid` and its latched abort. Whether it existed. Its parked waiters are
    /// the caller's to wake - see [`take_epoll_waiters`](Self::take_epoll_waiters).
    pub(super) fn epoll_destroy(&mut self, eid: i32) -> bool {
        let before = self.epolls.len();
        self.epolls.retain(|(id, _)| *id != eid);
        self.epoll_aborted.retain(|&e| e != eid);
        self.epolls.len() != before
    }

    pub(super) fn epoll_control(&mut self, eid: i32, op: i32, id: i32) -> bool {
        match self.epolls.iter_mut().find(|(e, _)| *e == eid) {
            Some((_, set)) => {
                match op {
                    3 => set.retain(|&s| s != id),
                    _ => {
                        if !set.contains(&id) {
                            set.push(id);
                        }
                    }
                }
                true
            }
            None => false,
        }
    }

    /// Record thread `thid` as parked in `sceNetEpollWait` on `eid`.
    pub(super) fn add_epoll_waiter(&mut self, thid: i32, eid: i32) {
        self.epoll_waiters.push((thid, eid));
    }

    /// Remove and return every thread parked on `eid`, in the order they parked.
    pub(super) fn take_epoll_waiters(&mut self, eid: i32) -> Vec<i32> {
        let woken: Vec<i32> = self
            .epoll_waiters
            .iter()
            .filter(|&&(_, e)| e == eid)
            .map(|&(t, _)| t)
            .collect();
        self.epoll_waiters.retain(|&(_, e)| e != eid);
        woken
    }

    /// Drop the epoll waits of threads whose timed `sceNetEpollWait` just ran out: they are no
    /// longer waiters an abort can interrupt.
    pub(super) fn forget_timed_out_epoll_waiters(&mut self, slept: &[i32]) {
        if !slept.is_empty() && !self.epoll_waiters.is_empty() {
            self.epoll_waiters.retain(|(t, _)| !slept.contains(t));
        }
    }
}
