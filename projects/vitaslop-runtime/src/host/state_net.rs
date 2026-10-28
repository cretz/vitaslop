use super::*;

// The offline network API the NID handlers call. The state and the logic are [`NetState`]'s
// (`host/net.rs`); every method here forwards, and composes in the scheduler where a call
// parks or wakes a thread.
impl VitaState {
    // --- SceNetAdhocMatching: the peer table that is always empty --------------

    /// Record the pool `sceNetAdhocMatchingInit` was given. `false` if it was already
    /// initialised, which the caller turns into ALREADY_INITIALIZED.
    #[inline]
    pub fn adhoc_matching_init(&mut self, pool_ptr: u32, pool_size: u32) -> bool {
        self.net.adhoc_matching_init(pool_ptr, pool_size)
    }

    /// Whether `sceNetAdhocMatchingInit` has been called.
    #[inline]
    pub fn adhoc_matching_ready(&self) -> bool {
        self.net.adhoc_matching_ready()
    }

    /// `sceNetAdhocMatchingTerm`: give the pool back and drop every context with it.
    /// `false` if the library was never initialised, which is the one way this can fail.
    ///
    /// The contexts go WITH the pool because they live in it: the memory the title lent
    /// the library at `Init` is the memory a context was carved from, so a term that left
    /// ids alive would leave them naming storage the title is free to reuse.
    #[inline]
    pub fn adhoc_matching_term(&mut self) -> bool {
        self.net.adhoc_matching_term()
    }

    /// Mint a matching context id. Ids count up from 1 in creation order, so they are a
    /// function of the guest's own call sequence and identical across runs.
    #[inline]
    pub fn adhoc_matching_create(&mut self) -> i32 {
        self.net.adhoc_matching_create()
    }

    /// Mark a context started/stopped. `None` if `id` names no live context.
    #[inline]
    pub fn adhoc_matching_set_started(&mut self, id: i32, started: bool) -> Option<()> {
        self.net.adhoc_matching_set_started(id, started)
    }

    /// Whether `id` names a live context.
    #[inline]
    pub fn adhoc_matching_live(&self, id: i32) -> bool {
        self.net.adhoc_matching_live(id)
    }

    /// Release a context. `false` if `id` names none.
    #[inline]
    pub fn adhoc_matching_delete(&mut self, id: i32) -> bool {
        self.net.adhoc_matching_delete(id)
    }

    // --- SceNet: the offline socket table ------------------------------------
    //
    // Real descriptors and real local state, no host sockets. See `vita::net` for what
    // is modelled and why. Everything here is a pure function of the guest's own call
    // sequence, so two runs of the same input produce identical socket ids.

    /// `sceNetSocket`: allocate a descriptor. Ids start above the file-descriptor range
    /// so a socket id mistakenly passed to `sceIoClose` cannot close a file.
    #[inline]
    pub fn net_socket(&mut self, name: &str, domain: i32, ty: i32, protocol: i32) -> i32 {
        self.net.socket(name, domain, ty, protocol)
    }

    #[inline]
    pub fn net_socket_exists(&self, id: i32) -> bool {
        self.net.socket_exists(id)
    }

    #[inline]
    pub fn net_close(&mut self, id: i32) -> bool {
        self.net.close(id)
    }

    #[inline]
    pub fn net_bind(&mut self, id: i32, ip: u32, port: u16) -> bool {
        self.net.bind(id, ip, port)
    }

    #[inline]
    pub fn net_listen(&mut self, id: i32) -> bool {
        self.net.listen(id)
    }

    #[inline]
    pub fn net_local_addr(&self, id: i32) -> Option<(u32, u16)> {
        self.net.local_addr(id)
    }

    #[inline]
    pub fn net_set_opt(&mut self, id: i32, level: i32, name: i32, value: u32) -> bool {
        self.net.set_opt(id, level, name, value)
    }

    /// An option never set reads back as 0, the kernel's default for every option a
    /// title queries here. `None` means the DESCRIPTOR is bad, which is a different
    /// answer from "the option is zero" and must not be conflated.
    #[inline]
    pub fn net_get_opt(&self, id: i32, level: i32, name: i32) -> Option<u32> {
        self.net.get_opt(id, level, name)
    }

    /// `sceNetShowNetstat`'s output: the live socket table, one line each.
    #[inline]
    pub fn net_netstat(&self) -> String {
        self.net.netstat()
    }

    /// The guest address of the CALLING thread's `sceNetErrnoLoc` slot, allocated on
    /// first use. Per thread, because two workers failing at once must not overwrite
    /// each other's reason - which is why the API hands back a location, not a value.
    pub fn net_errno_addr(&mut self) -> u32 {
        let thid = self.logical_thread(self.current);
        if let Some(addr) = self.net.errno_slot(thid) {
            return addr;
        }
        let addr = self.galloc(4, 4);
        self.net.set_errno_slot(thid, addr);
        addr
    }

    /// Record why the last SceNet call failed, where `sceNetErrnoLoc` will read it.
    pub fn net_set_errno(&mut self, code: i32) {
        let addr = self.net_errno_addr();
        if addr != 0 {
            self.pending_stat_writes.push((addr, code as u32));
        }
    }

    // --- SceHttp: the offline object graph -----------------------------------
    //
    // Real objects and real local state, no host connections. See `vita::http`.

    /// Allocate a template/connection/request. The caller has already checked that
    /// `parent` is a live object of the level above, so this only allocates.
    #[inline]
    pub fn http_create(&mut self, kind: HttpKind, parent: i32, label: &str, method: i32) -> i32 {
        self.net.http_create(kind, parent, label, method)
    }

    /// Turn one of this object's [`HttpSwitch`] settings on or off. `false` if `id` names
    /// no live object.
    #[inline]
    pub fn http_set_switch(&mut self, id: i32, which: HttpSwitch, on: bool) -> bool {
        self.net.http_set_switch(id, which, on)
    }

    /// ...and read it back. `None` if `id` names no live object.
    #[inline]
    pub fn http_switch(&self, id: i32, which: HttpSwitch) -> Option<bool> {
        self.net.http_switch(id, which)
    }

    /// `sceHttpsEnableOption` / `sceHttpsDisableOption`: the guest turned these
    /// `SCE_HTTPS_FLAG_*` bits on or off library-wide. The newest call wins per bit.
    #[inline]
    pub fn https_set_options(&mut self, flags: u32, on: bool) {
        self.net.https_set_options(flags, on)
    }

    /// The `SCE_HTTPS_FLAG_*` bits the guest turned `(on, off)`. Kept as the two sets it
    /// named rather than one word over an assumed default: no handshake ever happens to
    /// consume them, and a trace should show what the title asked for.
    #[inline]
    pub fn https_options(&self) -> (u32, u32) {
        self.net.https_options()
    }

    /// Is `id` a live object at all, of any kind?
    #[inline]
    pub fn http_exists(&self, id: i32) -> bool {
        self.net.http_exists(id)
    }

    /// Is `id` a live object of exactly `kind`?
    #[inline]
    pub fn http_is(&self, id: i32, kind: HttpKind) -> bool {
        self.net.http_is(id, kind)
    }

    /// Free `id`. Deleting a template or connection also frees everything under it,
    /// which is what the library does - the children cannot outlive the object that
    /// owns their settings, and leaving them live would let a request survive its
    /// connection and report a different error than hardware gives.
    #[inline]
    pub fn http_delete(&mut self, id: i32) -> bool {
        self.net.http_delete(id)
    }

    /// Record a connect/send/receive timeout (`which` is 0/1/2) against a live object.
    #[inline]
    pub fn http_set_timeout(&mut self, id: i32, which: usize, usec: u32) -> bool {
        self.net.http_set_timeout(id, which, usec)
    }

    /// Record a request header against a live object.
    #[inline]
    pub fn http_add_header(&mut self, id: i32, name: &str, value: &str) -> bool {
        self.net.http_add_header(id, name, value)
    }

    /// One line per live SceHttp object, for a shutdown report or a trace: the graph
    /// the title built, and what it asked for over it.
    #[inline]
    pub fn http_report(&self) -> Vec<String> {
        self.net.http_report()
    }

    #[inline]
    pub fn net_resolver_create(&mut self) -> i32 {
        self.net.resolver_create()
    }

    #[inline]
    pub fn net_resolver_destroy(&mut self, rid: i32) -> bool {
        self.net.resolver_destroy(rid)
    }

    #[inline]
    pub fn net_resolver_set_error(&mut self, rid: i32, code: i32) {
        self.net.resolver_set_error(rid, code)
    }

    #[inline]
    pub fn net_resolver_error(&self, rid: i32) -> Option<i32> {
        self.net.resolver_error(rid)
    }

    #[inline]
    pub fn net_epoll_create(&mut self) -> i32 {
        self.net.epoll_create()
    }

    /// Latch an abort on `eid` for its next wait. `false` if `eid` names no epoll.
    #[inline]
    pub fn net_epoll_latch_abort(&mut self, eid: i32) -> bool {
        self.net.epoll_latch_abort(eid)
    }

    /// Consume a latched abort on `eid`, if there is one.
    #[inline]
    pub fn net_epoll_take_abort(&mut self, eid: i32) -> bool {
        self.net.epoll_take_abort(eid)
    }

    #[inline]
    pub fn net_epoll_exists(&self, eid: i32) -> bool {
        self.net.epoll_exists(eid)
    }

    pub fn net_epoll_destroy(&mut self, eid: i32) -> bool {
        let existed = self.net.epoll_destroy(eid);
        self.net_epoll_wake(eid, crate::vita::net::NET_FAIL as u32);
        existed
    }

    /// Park the current thread in `sceNetEpollWait` on `eid`: for `timeout_us` of guest time
    /// (it then returns the 0 events already in its `r0`), or with `None` until an abort or a
    /// destroy wakes it.
    pub fn net_epoll_park(&mut self, eid: i32, timeout_us: Option<u64>) {
        self.net.add_epoll_waiter(self.current, eid);
        if let Some(us) = timeout_us {
            self.sleep_park(us);
        }
    }

    /// Wake every thread waiting on `eid`, each returning `code`. Returns how many woke.
    pub fn net_epoll_wake(&mut self, eid: i32, code: u32) -> usize {
        let woken = self.net.take_epoll_waiters(eid);
        for &thid in &woken {
            self.sleep_waiters.retain(|&(t, _)| t != thid);
            self.pending_resume_codes.push((thid, code));
            push_wake!(self, thid);
        }
        woken.len()
    }

    /// `SCE_NET_EPOLL_CTL_ADD` (1) / `MOD` (2) / `DEL` (3) over the registered set.
    #[inline]
    pub fn net_epoll_control(&mut self, eid: i32, op: i32, id: i32) -> bool {
        self.net.epoll_control(eid, op, id)
    }
}
