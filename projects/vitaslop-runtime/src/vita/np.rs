//! The NP utility surface a console with no PSN sign-in still runs locally - NpCommon's id
//! helpers, NpBasic's cache queries, NpCommerce2's store icon and request bookkeeping, the
//! Lookup and WordFilter object models - and SceNearUtil, the location-based gift service.
//!
//! # Local work is real; the network step reports that there is none
//! Comparing two NP ids, tagging one with a platform, building title contexts and requests,
//! keeping a gift: all of that is the guest's own state and is done for real, with the id and
//! limit checks the libraries make. What needs a server - resolving an online id, fetching a
//! profile, sanitising a comment - fails with the community "not logged in" code
//! (`NO_LOGIN`), synchronously for the synchronous calls and as the completed result of an
//! asynchronous one. The near service never discovers anything: nobody is near, no gift has
//! arrived, and the gift this title sets is kept but never travels.

use crate::host::{GuestCtx, Ptr, VitaState};
use crate::hostcall;

// --- Error codes ---------------------------------------------------------------------------

const NP_ERROR_INVALID_ARGUMENT: i32 = 0x8055_0003u32 as i32;
const NP_ERROR_UNKNOWN_PLATFORM_TYPE: i32 = 0x8055_0004u32 as i32;
const NP_UTIL_ERROR_INVALID_ARGUMENT: i32 = 0x8055_0601u32 as i32;
const NP_UTIL_ERROR_NOT_MATCH: i32 = 0x8055_0609u32 as i32;

const COMMUNITY_ERROR_ALREADY_INITIALIZED: i32 = 0x8055_0701u32 as i32;
const COMMUNITY_ERROR_NOT_INITIALIZED: i32 = 0x8055_0702u32 as i32;
/// `SCE_NP_COMMUNITY_ERROR_NO_LOGIN`: the community libraries' (score, TUS, lookup) "not signed
/// in" - the only sign-in code in their block.
pub(super) const COMMUNITY_ERROR_NO_LOGIN: i32 = 0x8055_0705u32 as i32;
const COMMUNITY_ERROR_TOO_MANY_OBJECTS: i32 = 0x8055_0706u32 as i32;
const COMMUNITY_ERROR_ABORTED: i32 = 0x8055_0707u32 as i32;
const COMMUNITY_ERROR_INSUFFICIENT_ARGUMENT: i32 = 0x8055_070Cu32 as i32;
const COMMUNITY_ERROR_INVALID_ID: i32 = 0x8055_070Eu32 as i32;
const COMMUNITY_ERROR_INVALID_TYPE: i32 = 0x8055_0711u32 as i32;

const BASIC_ERROR_INVALID_ARGUMENT: i32 = 0x8055_1D02u32 as i32;
/// `SCE_NP_BASIC_ERROR_SIGNED_OUT`.
pub(super) const BASIC_ERROR_SIGNED_OUT: i32 = 0x8055_1D06u32 as i32;

const COMMERCE2_ERROR_INVALID_ARGUMENT: i32 = 0x8055_0F03u32 as i32;
const COMMERCE2_ERROR_REQ_NOT_FOUND: i32 = 0x8055_0F0Eu32 as i32;
/// `SCE_NP_COMMERCE2_ERROR_NOT_SIGNIN`: "not signed in to PlayStation Network".
pub(super) const COMMERCE2_ERROR_NOT_SIGNIN: i32 = 0x8055_0F72u32 as i32;

/// The SceNpPartyGameUtil code (block `0x805525xx`) for "there is no party to describe". NO
/// LOCAL OR PUBLISHED SOURCE NAMES IT: the name is this module's, and the value and meaning are
/// read off two titles' own code, which test the member queries' return against exactly this
/// value and take a no-party path on it (one maps it to a "not in a party" status). Signed out
/// the console cannot be in a party - the system party runs over PSN - so every party query
/// answers with it. See the party group in `vita::dispatch`.
pub(super) const PARTY_ERROR_NOT_IN_PARTY: i32 = 0x8055_2506u32 as i32;

const NEAR_ERROR_INVALID_ARGUMENT: i32 = 0x8010_4901u32 as i32;
const NEAR_ERROR_ALREADY_INITIALIZED: i32 = 0x8010_4903u32 as i32;
const NEAR_ERROR_NO_MEMORY: i32 = 0x8010_4904u32 as i32;
const NEAR_ERROR_NOT_INITIALIZED: i32 = 0x8010_4905u32 as i32;
const NEAR_ERROR_RETRIEVE_FAILED: i32 = 0x8010_4912u32 as i32;
const NEAR_ERROR_DELETION_FAILED: i32 = 0x8010_4918u32 as i32;

// --- SceNpCommon -----------------------------------------------------------------------------

/// `SceNpId` (0x24 bytes): the online id's 16 characters, its terminator and padding, then
/// `opt[8]` - four account bytes and a four-byte platform tag - and `reserved[8]`.
const NPID_SIZE: usize = 0x24;
const NPID_ONLINE_ID_LEN: usize = 16;
const NPID_OPT: usize = 0x14;
const NPID_PLATFORM: usize = 0x18;

/// The platform tag `sceNpSetPlatformType` writes for each `SceNpPlatformType`.
fn platform_tag(kind: i32) -> Option<[u8; 4]> {
    match kind {
        0 => Some([0; 4]),
        1 => Some(*b"ps3\0"),
        2 => Some(*b"psp2"),
        3 => Some(*b"ps4\0"),
        _ => None,
    }
}

/// The comparison `sceNpCmpNpId` makes: the online ids as C strings over at most 16 bytes,
/// byte-exact; then the four account bytes; then the platform tags, which are equal when
/// neither id carries one.
fn npids_match(a: &[u8], b: &[u8]) -> bool {
    let name = |id: &[u8]| {
        let raw = &id[..NPID_ONLINE_ID_LEN];
        raw[..raw.iter().position(|&c| c == 0).unwrap_or(raw.len())].to_vec()
    };
    if name(a) != name(b) || a[NPID_OPT..NPID_PLATFORM] != b[NPID_OPT..NPID_PLATFORM] {
        return false;
    }
    (a[NPID_PLATFORM] == 0 && b[NPID_PLATFORM] == 0)
        || a[NPID_PLATFORM..NPID_PLATFORM + 4] == b[NPID_PLATFORM..NPID_PLATFORM + 4]
}

/// int sceNpCmpNpId(const SceNpId *npid1, const SceNpId *npid2)
///
/// 0 for the same account, NOT_MATCH for different ones. Purely local, signed in or not.
#[hostcall]
pub(super) fn cmp_np_id(ctx: &mut GuestCtx, _st: &mut VitaState, a: Ptr, b: Ptr) -> i32 {
    if a.is_null() || b.is_null() {
        NP_UTIL_ERROR_INVALID_ARGUMENT
    } else if npids_match(
        &ctx.read_bytes(a.addr(), NPID_SIZE),
        &ctx.read_bytes(b.addr(), NPID_SIZE),
    ) {
        0
    } else {
        NP_UTIL_ERROR_NOT_MATCH
    }
}

/// int sceNpSetPlatformType(SceNpId *npId, int platformType)
///
/// Write the platform tag (+0x18) for NONE / PS3 / VITA / PS4; nothing else in the id changes.
#[hostcall]
pub(super) fn set_platform_type(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    id: Ptr,
    kind: i32,
) -> i32 {
    match (id.is_null(), platform_tag(kind)) {
        (true, _) => NP_ERROR_INVALID_ARGUMENT,
        (false, None) => NP_ERROR_UNKNOWN_PLATFORM_TYPE,
        (false, Some(tag)) => {
            ctx.write_bytes(id.addr() + NPID_PLATFORM as u32, &tag);
            0
        }
    }
}

// --- SceNpBasic ------------------------------------------------------------------------------
//
// The block list and friends' presence are read from the cache the system fills from PSN once
// signed in. Signed out, that cache is empty: no one is blocked and there is no friend whose
// presence could be read.

/// int sceNpBasicGetBlockListEntryCount(SceUInt32 *count)
#[hostcall]
pub(super) fn basic_get_block_list_entry_count(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    count: Ptr,
) -> i32 {
    if count.is_null() {
        BASIC_ERROR_INVALID_ARGUMENT
    } else {
        ctx.write_u32(count.addr(), 0);
        0
    }
}

/// int sceNpBasicGetBlockListEntries(SceUInt32 startIndex, SceNpId *entries,
///                                   SceUInt32 numEntries, SceUInt32 *retrieved)
#[hostcall]
pub(super) fn basic_get_block_list_entries(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    _start: u32,
    entries: Ptr,
    _num: u32,
    retrieved: Ptr,
) -> i32 {
    if entries.is_null() || retrieved.is_null() {
        BASIC_ERROR_INVALID_ARGUMENT
    } else {
        ctx.write_u32(retrieved.addr(), 0);
        0
    }
}

/// int sceNpBasicGetGamePresenceOfFriend(const SceNpId *friendId, SceNpBasicGamePresence *p)
///
/// Signed out there are no friends, so no presence to give; `*p` is left as it was.
#[hostcall]
pub(super) fn basic_get_game_presence_of_friend(
    _st: &mut VitaState,
    friend: Ptr,
    presence: Ptr,
) -> i32 {
    if friend.is_null() || presence.is_null() {
        BASIC_ERROR_INVALID_ARGUMENT
    } else {
        BASIC_ERROR_SIGNED_OUT
    }
}

/// int sceNpBasicRegisterJoinGameAckHandler(SceNpBasicJoinGameAckHandler handler, void *userdata)
///
/// The handler a friend's request to join this title's session is delivered to. Registering
/// it is local and succeeds; signed out no friend exists to ask, so it is never called. A
/// null handler is the caller's error.
#[hostcall]
pub(super) fn basic_register_join_game_ack_handler(
    st: &mut VitaState,
    handler: u32,
    userdata: u32,
) -> i32 {
    if handler == 0 {
        BASIC_ERROR_INVALID_ARGUMENT
    } else {
        st.np.join_game_ack_handler = Some((handler, userdata));
        0
    }
}

/// int sceNpBasicUnregisterJoinGameAckHandler(void)
///
/// Drops the handler. Succeeds with none registered too: a title unwinding its online setup
/// must not be handed an error on the way out (see the teardown group in `vita::dispatch`).
#[hostcall]
pub(super) fn basic_unregister_join_game_ack_handler(st: &mut VitaState) -> i32 {
    let _ = st.np.join_game_ack_handler.take();
    0
}

// --- SceNpMatching2: the two calls that need no context ---------------------------------------
//
// Everything else in matching2 hangs off a context only a signed-in account can create, and
// answers NOT_NP_SIGN_IN (see the group in `vita::dispatch`). These two do not: one reads back
// the pool the library was initialised with, the other asks for this console's own addresses.
// The codes are matching2's and signaling's own blocks (0x80550Cxx / 0x80550Exx) - from a
// public Vita error-name list, since the local wiki has no matching2 block.

const MATCHING2_ERROR_ALREADY_INITIALIZED: i32 = 0x8055_0C02u32 as i32;
const MATCHING2_ERROR_NOT_INITIALIZED: i32 = 0x8055_0C03u32 as i32;
const MATCHING2_ERROR_INVALID_ARGUMENT: i32 = 0x8055_0C0Au32 as i32;
const MATCHING2_SIGNALING_ERROR_NOT_INITIALIZED: i32 = 0x8055_0E01u32 as i32;
/// `SCE_NP_MATCHING2_SIGNALING_ERROR_NETINFO_NOT_AVAILABLE`: signaling cannot say what this
/// console's addresses are - which, with no interface up, it cannot.
const MATCHING2_SIGNALING_ERROR_NETINFO_NOT_AVAILABLE: i32 = 0x8055_0E0Bu32 as i32;

/// int sceNpMatching2Init(SceSize poolSize, int threadPriority, int cpuAffinityMask,
///                        SceSize threadStackSize)
///
/// The library's pool is recorded (see [`matching2_get_memory_info`]); a second init while one
/// is live is the library's ALREADY_INITIALIZED.
#[hostcall]
pub(super) fn matching2_init(st: &mut VitaState, pool_size: u32) -> i32 {
    if st.np.matching2_pool.is_some() {
        MATCHING2_ERROR_ALREADY_INITIALIZED
    } else {
        st.np.matching2_pool = Some(pool_size);
        0
    }
}

/// int sceNpMatching2Term(void)
///
/// Releases the pool. Succeeds with no init behind it too, as the rest of the online stack's
/// teardown does.
#[hostcall]
pub(super) fn matching2_term(st: &mut VitaState) -> i32 {
    st.np.matching2_pool = None;
    0
}

/// int sceNpMatching2GetMemoryInfo(SceNpMatching2MemoryInfo *memInfo)
///
/// `{ totalMemSize, curMemUsage, maxMemUsage }`: the pool `Init` was handed, and how much of it
/// is used now and at most. None of it, ever - this engine keeps the library's state on the
/// host and no context or request is created to take any - so both usage figures are a true 0.
#[hostcall]
pub(super) fn matching2_get_memory_info(ctx: &mut GuestCtx, st: &mut VitaState, info: Ptr) -> i32 {
    match st.np.matching2_pool {
        None => MATCHING2_ERROR_NOT_INITIALIZED,
        Some(_) if info.is_null() => MATCHING2_ERROR_INVALID_ARGUMENT,
        Some(pool) => {
            for (i, v) in [pool, 0, 0].iter().enumerate() {
                ctx.write_u32(info.addr() + 4 * i as u32, *v);
            }
            0
        }
    }
}

/// int sceNpMatching2SignalingGetLocalNetInfo(SceNpMatching2SignalingNetInfo *netinfo)
///
/// This console's local and NAT-mapped addresses as signaling sees them. It needs no context
/// and no account - only a network, and there is none up: the interface `sceNetCtl` reports
/// disconnected has no address to give, so the answer is signaling's own
/// NETINFO_NOT_AVAILABLE and `*netinfo` is left as the caller had it. Before `Init` it is
/// signaling's NOT_INITIALIZED.
#[hostcall]
pub(super) fn matching2_signaling_get_local_net_info(st: &mut VitaState, _netinfo: Ptr) -> i32 {
    if st.np.matching2_pool.is_none() {
        MATCHING2_SIGNALING_ERROR_NOT_INITIALIZED
    } else {
        MATCHING2_SIGNALING_ERROR_NETINFO_NOT_AVAILABLE
    }
}

// --- SceNpCommerce2 --------------------------------------------------------------------------

/// int sceNpCommerce2AbortReq(SceUInt32 reqId)
///
/// No request is ever accepted signed out (the session they hang off is refused), so every id
/// names nothing.
#[hostcall]
pub(super) fn commerce2_abort_req(_st: &mut VitaState, _req: u32) -> i32 {
    COMMERCE2_ERROR_REQ_NOT_FOUND
}

/// int sceNpCommerce2DestroyGetProductInfoListResult(SceNpCommerce2GetProductInfoListResult *r)
///
/// Releases what initialising the result set up. No fetch can have filled one, so there is
/// nothing held; the call succeeds for any result object.
#[hostcall]
pub(super) fn commerce2_destroy_product_info_list_result(_st: &mut VitaState, result: Ptr) -> i32 {
    if result.is_null() {
        COMMERCE2_ERROR_INVALID_ARGUMENT
    } else {
        0
    }
}

/// int sceNpCommerce2ShowPsStoreIcon(int position)
///
/// Ask the system to overlay the PS Store button hint at the bottom centre (0), left (1) or
/// right (2) of the game's picture. Recorded as the overlay state; this front end draws no
/// system overlays.
#[hostcall]
pub(super) fn commerce2_show_ps_store_icon(st: &mut VitaState, position: i32) -> i32 {
    if (0..=2).contains(&position) {
        st.np.store_icon = Some(position);
        0
    } else {
        COMMERCE2_ERROR_INVALID_ARGUMENT
    }
}

/// int sceNpCommerce2HidePsStoreIcon(void)
#[hostcall]
pub(super) fn commerce2_hide_ps_store_icon(st: &mut VitaState) -> i32 {
    st.np.store_icon = None;
    0
}

// --- Lookup and WordFilter: the community object model ---------------------------------------

/// At most this many title contexts, and this many requests, at once.
const COMMUNITY_MAX_OBJECTS: usize = 32;

#[derive(Default)]
struct Request {
    id: i32,
    ctx: i32,
    aborted: bool,
    /// The code the operation this request carried finished with; `None` until one is issued.
    result: Option<i32>,
}

/// One service's title contexts and requests (Lookup and WordFilter keep separate ones).
/// Ids start at 1 and are never reused within a run.
#[derive(Default)]
pub struct Community {
    initialized: bool,
    ctxs: Vec<i32>,
    reqs: Vec<Request>,
    next_ctx: i32,
    next_req: i32,
}

/// Which community service a call is for.
#[derive(Clone, Copy)]
pub(super) enum Service {
    Lookup,
    WordFilter,
}

fn community(st: &mut VitaState, s: Service) -> &mut Community {
    match s {
        Service::Lookup => &mut st.np.lookup,
        Service::WordFilter => &mut st.np.word_filter,
    }
}

impl Community {
    fn init(&mut self) -> i32 {
        if self.initialized {
            return COMMUNITY_ERROR_ALREADY_INITIALIZED;
        }
        self.initialized = true;
        0
    }

    fn term(&mut self) -> i32 {
        if !self.initialized {
            return COMMUNITY_ERROR_NOT_INITIALIZED;
        }
        self.initialized = false;
        self.ctxs.clear();
        self.reqs.clear();
        0
    }

    fn create_ctx(&mut self, comm_id_given: bool) -> i32 {
        if !self.initialized {
            COMMUNITY_ERROR_NOT_INITIALIZED
        } else if !comm_id_given {
            COMMUNITY_ERROR_INSUFFICIENT_ARGUMENT
        } else if self.ctxs.len() >= COMMUNITY_MAX_OBJECTS {
            COMMUNITY_ERROR_TOO_MANY_OBJECTS
        } else {
            self.next_ctx += 1;
            self.ctxs.push(self.next_ctx);
            self.next_ctx
        }
    }

    /// Deleting a context deletes its requests with it.
    fn delete_ctx(&mut self, id: i32) -> i32 {
        if !self.initialized {
            return COMMUNITY_ERROR_NOT_INITIALIZED;
        }
        let before = self.ctxs.len();
        self.ctxs.retain(|&c| c != id);
        if self.ctxs.len() == before {
            return COMMUNITY_ERROR_INVALID_ID;
        }
        self.reqs.retain(|r| r.ctx != id);
        0
    }

    fn create_req(&mut self, ctx: i32) -> i32 {
        if !self.initialized {
            COMMUNITY_ERROR_NOT_INITIALIZED
        } else if !self.ctxs.contains(&ctx) {
            COMMUNITY_ERROR_INVALID_ID
        } else if self.reqs.len() >= COMMUNITY_MAX_OBJECTS {
            COMMUNITY_ERROR_TOO_MANY_OBJECTS
        } else {
            self.next_req += 1;
            self.reqs.push(Request {
                id: self.next_req,
                ctx,
                ..Default::default()
            });
            self.next_req
        }
    }

    fn req(&mut self, id: i32) -> Result<&mut Request, i32> {
        if !self.initialized {
            return Err(COMMUNITY_ERROR_NOT_INITIALIZED);
        }
        self.reqs
            .iter_mut()
            .find(|r| r.id == id)
            .ok_or(COMMUNITY_ERROR_INVALID_ID)
    }

    fn delete_req(&mut self, id: i32) -> i32 {
        match self.req(id) {
            Err(e) => e,
            Ok(_) => {
                self.reqs.retain(|r| r.id != id);
                0
            }
        }
    }

    /// The request stays allocated; its operation, in flight or still to be issued, ends ABORTED.
    fn abort_req(&mut self, id: i32) -> i32 {
        match self.req(id) {
            Err(e) => e,
            Ok(r) => {
                r.aborted = true;
                0
            }
        }
    }

    /// Issue the request's one operation. It needs the server, which is unreachable signed out,
    /// so it ends NO_LOGIN; a request already used, or aborted, is refused first.
    fn issue(&mut self, id: i32) -> Result<i32, i32> {
        let r = self.req(id)?;
        if r.aborted {
            return Err(COMMUNITY_ERROR_ABORTED);
        }
        if r.result.is_some() {
            return Err(COMMUNITY_ERROR_INVALID_TYPE);
        }
        r.result = Some(COMMUNITY_ERROR_NO_LOGIN);
        Ok(COMMUNITY_ERROR_NO_LOGIN)
    }

    /// Poll or wait: the operation has finished (it completes at once), so `Ok(code)` is the
    /// code to write to `*result`. A request with no operation issued has nothing to report.
    fn completion(&mut self, id: i32) -> Result<i32, i32> {
        let r = self.req(id)?;
        match (r.result, r.aborted) {
            (_, true) => Ok(COMMUNITY_ERROR_ABORTED),
            (Some(code), false) => Ok(code),
            (None, false) => Err(COMMUNITY_ERROR_INVALID_ID),
        }
    }
}

/// int sceNpLookupInit / sceNpWordFilterInit(int, int threadPriority, int, int)
pub(super) fn community_init(ctx: &mut GuestCtx, st: &mut VitaState, s: Service) {
    let r = community(st, s).init();
    ctx.ret(r as u32);
}

/// int sceNpLookupTerm / sceNpWordFilterTerm(void)
pub(super) fn community_term(ctx: &mut GuestCtx, st: &mut VitaState, s: Service) {
    let r = community(st, s).term();
    ctx.ret(r as u32);
}

/// int sceNp{Lookup,WordFilter}CreateTitleCtx(const SceNpCommunicationId *commId,
///                                            const SceNpId *selfNpId)
pub(super) fn community_create_ctx(ctx: &mut GuestCtx, st: &mut VitaState, s: Service) {
    let comm = ctx.arg(0) != 0;
    let r = community(st, s).create_ctx(comm);
    ctx.ret(r as u32);
}

/// int sceNp{Lookup,WordFilter}DeleteTitleCtx(int titleCtxId)
pub(super) fn community_delete_ctx(ctx: &mut GuestCtx, st: &mut VitaState, s: Service) {
    let id = ctx.arg(0) as i32;
    let r = community(st, s).delete_ctx(id);
    ctx.ret(r as u32);
}

/// int sceNp{Lookup,WordFilter}CreateRequest(int titleCtxId)
pub(super) fn community_create_req(ctx: &mut GuestCtx, st: &mut VitaState, s: Service) {
    let id = ctx.arg(0) as i32;
    let r = community(st, s).create_req(id);
    ctx.ret(r as u32);
}

/// int sceNp{Lookup,WordFilter}DeleteRequest(int reqId)
pub(super) fn community_delete_req(ctx: &mut GuestCtx, st: &mut VitaState, s: Service) {
    let id = ctx.arg(0) as i32;
    let r = community(st, s).delete_req(id);
    ctx.ret(r as u32);
}

/// int sceNp{Lookup,WordFilter}AbortRequest(int reqId)
pub(super) fn community_abort_req(ctx: &mut GuestCtx, st: &mut VitaState, s: Service) {
    let id = ctx.arg(0) as i32;
    let r = community(st, s).abort_req(id);
    ctx.ret(r as u32);
}

/// int sceNpLookupNpId(int reqId, const SceNpOnlineId *onlineId, SceNpId *npId, void *option)
/// int sceNpLookupUserProfile(int reqId, ...)
///
/// The synchronous forms: the lookup needs the server, so the call returns NO_LOGIN and writes
/// none of its outputs.
pub(super) fn community_sync_op(ctx: &mut GuestCtx, st: &mut VitaState, s: Service) {
    let id = ctx.arg(0) as i32;
    let r = community(st, s).issue(id).unwrap_or_else(|e| e);
    ctx.ret(r as u32);
}

/// int sceNpLookupNpIdAsync / sceNpLookupUserProfileAsync(int reqId, ...)
/// int sceNpWordFilterSanitizeCommentAsync(int reqId, const char *comment, char *out, void *opt)
///
/// The asynchronous forms start (0) and complete at once with NO_LOGIN, which the poll or wait
/// then reports. Nothing is written: the comment buffer, which may also be the input, is left
/// exactly as the title passed it.
pub(super) fn community_async_op(ctx: &mut GuestCtx, st: &mut VitaState, s: Service) {
    let id = ctx.arg(0) as i32;
    let r = match community(st, s).issue(id) {
        Ok(_) => 0,
        Err(e) => e,
    };
    ctx.ret(r as u32);
}

/// int sceNp{Lookup,WordFilter}PollAsync(int reqId, int *result)
/// int sceNpLookupWaitAsync(int reqId, int *result)
///
/// 0 = finished, with the operation's code in `*result`. The operation never takes time, so a
/// poll and a wait answer the same.
pub(super) fn community_completion(ctx: &mut GuestCtx, st: &mut VitaState, s: Service) {
    let (id, out) = (ctx.arg(0) as i32, ctx.arg(1));
    let r = match community(st, s).completion(id) {
        Ok(code) => {
            if out != 0 {
                ctx.write_u32(out, code as u32);
            }
            0
        }
        Err(e) => e,
    };
    ctx.ret(r as u32);
}

// --- SceNearUtil -------------------------------------------------------------------------------

/// `SceNearGiftInfo`, the gift's descriptive record.
const NEAR_GIFT_INFO_SIZE: usize = 0x1A0;
/// The condition record that travels with a gift (`sceNearSetGift`'s last argument,
/// `sceNearGetGift`'s 0x2C-byte area).
const NEAR_GIFT_CONDITION_SIZE: usize = 0x2C;
/// The largest image or data blob kept for a gift. Larger blobs are stored up to this; the
/// title's own caps are far below it.
const NEAR_GIFT_BLOB_MAX: usize = 1 << 20;

/// The one gift this title has set: kept for `sceNearGetGift`, never distributed.
struct NearGift {
    id: u32,
    info: Vec<u8>,
    image: Vec<u8>,
    data: Vec<u8>,
    param: u32,
    condition: Vec<u8>,
}

#[derive(Default)]
pub struct Near {
    initialized: bool,
    gift: Option<NearGift>,
}

/// Every near call but `sceNearInitialize` needs the utility initialised first.
fn near_ready(st: &VitaState) -> Result<(), i32> {
    if st.np.near.initialized {
        Ok(())
    } else {
        Err(NEAR_ERROR_NOT_INITIALIZED)
    }
}

/// int sceNearInitialize(const SceNpCommunicationId *commId, const SceNearWorkArea *work, int)
///
/// `work` is `{ void *base; SceSize size; }`, the memory the utility runs in. Initialising talks
/// to the local near process only, so it succeeds offline; a second call says so.
#[hostcall]
pub(super) fn near_initialize(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    comm: Ptr,
    work: Ptr,
    _version: i32,
) -> i32 {
    if comm.is_null() || work.is_null() {
        NEAR_ERROR_INVALID_ARGUMENT
    } else if st.np.near.initialized {
        NEAR_ERROR_ALREADY_INITIALIZED
    } else if ctx.read_u32(work.addr()) == 0 || ctx.read_u32(work.addr() + 4) == 0 {
        NEAR_ERROR_NO_MEMORY
    } else {
        st.np.near.initialized = true;
        0
    }
}

fn near_finalize_in(st: &mut VitaState) -> i32 {
    match near_ready(st) {
        Err(e) => e,
        Ok(()) => {
            st.np.near.initialized = false;
            0
        }
    }
}

/// int sceNearFinalize(const SceNpCommunicationId *commId)
///
/// The gift the title set stays with the near service, as it does on hardware.
#[hostcall]
pub(super) fn near_finalize(st: &mut VitaState, _comm: Ptr) -> i32 {
    near_finalize_in(st)
}

/// int sceNearRefresh(const SceNpCommunicationId *commId)
///
/// Re-read what has been discovered - which is nothing.
#[hostcall]
pub(super) fn near_refresh(st: &mut VitaState, _comm: Ptr) -> i32 {
    near_ready(st).err().unwrap_or(0)
}

/// int sceNearSetGift(SceUInt32 giftId, const SceNearGiftInfo *info,
///                    SceSize imageSize, const void *image, SceSize dataSize, const void *data,
///                    SceUInt32 param, const void *condition)
///
/// Register this title's gift, replacing any it set before. Kept so `sceNearGetGift` hands it
/// back and `sceNearDeleteGift` removes it; with nobody near, it is never passed on.
#[hostcall]
#[allow(clippy::too_many_arguments)]
pub(super) fn near_set_gift(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    id: u32,
    info: Ptr,
    image_size: u32,
    image: Ptr,
    data_size: u32,
    data: Ptr,
    param: u32,
    condition: Ptr,
) -> i32 {
    let blob = |ctx: &GuestCtx, p: Ptr, n: u32| {
        if p.is_null() {
            Vec::new()
        } else {
            ctx.read_bytes(p.addr(), (n as usize).min(NEAR_GIFT_BLOB_MAX))
        }
    };
    match near_ready(st) {
        Err(e) => e,
        Ok(())
            if info.is_null()
                || (image.is_null() && image_size != 0)
                || (data.is_null() && data_size != 0) =>
        {
            NEAR_ERROR_INVALID_ARGUMENT
        }
        Ok(()) => {
            let gift = NearGift {
                id,
                info: ctx.read_bytes(info.addr(), NEAR_GIFT_INFO_SIZE),
                image: blob(ctx, image, image_size),
                data: blob(ctx, data, data_size),
                param,
                condition: if condition.is_null() {
                    vec![0; NEAR_GIFT_CONDITION_SIZE]
                } else {
                    ctx.read_bytes(condition.addr(), NEAR_GIFT_CONDITION_SIZE)
                },
            };
            st.np.near.gift = Some(gift);
            0
        }
    }
}

/// int sceNearGetGift(SceUInt32 *giftId, SceNearGiftInfo *info, SceSize *imageSize, void *image,
///                    SceSize *dataSize, void *data, SceUInt32 *param, void *condition)
///
/// 0 when this title has no gift set. With one, 1 and every output filled: each blob is copied
/// up to the capacity the caller put in its size word, and that word then holds the size copied.
#[hostcall]
#[allow(clippy::too_many_arguments)]
pub(super) fn near_get_gift(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    id: Ptr,
    info: Ptr,
    image_size: Ptr,
    image: Ptr,
    data_size: Ptr,
    data: Ptr,
    param: Ptr,
    condition: Ptr,
) -> i32 {
    match near_ready(st) {
        Err(e) => e,
        Ok(()) if id.is_null() || info.is_null() => NEAR_ERROR_INVALID_ARGUMENT,
        Ok(()) => match st.np.near.gift.as_ref() {
            None => 0,
            Some(g) => {
                ctx.write_u32(id.addr(), g.id);
                ctx.write_bytes(info.addr(), &g.info);
                for (size, buf, blob) in [(image_size, image, &g.image), (data_size, data, &g.data)]
                {
                    if size.is_null() {
                        continue;
                    }
                    let n = if buf.is_null() {
                        0
                    } else {
                        (ctx.read_u32(size.addr()) as usize).min(blob.len())
                    };
                    if n > 0 {
                        ctx.write_bytes(buf.addr(), &blob[..n]);
                    }
                    ctx.write_u32(size.addr(), n as u32);
                }
                if !param.is_null() {
                    ctx.write_u32(param.addr(), g.param);
                }
                if !condition.is_null() {
                    ctx.write_bytes(condition.addr(), &g.condition);
                }
                1
            }
        },
    }
}

/// int sceNearDeleteGift(SceUInt32 giftId)
#[hostcall]
pub(super) fn near_delete_gift(st: &mut VitaState, id: u32) -> i32 {
    match near_ready(st) {
        Err(e) => e,
        Ok(()) if st.np.near.gift.as_ref().is_some_and(|g| g.id == id) => {
            st.np.near.gift = None;
            0
        }
        Ok(()) => NEAR_ERROR_DELETION_FAILED,
    }
}

/// int sceNearGetDiscoveredGifts(SceUInt32 *ioCount, SceNearGiftIdList *list)
/// int sceNearGetNeighbors(SceUInt32 *ioCount, SceNearNeighborList *list)
///
/// The list calls: the return value is how many there are, and the count word is set to how many
/// were written. Nothing has been discovered and nobody is near, so both are 0.
#[hostcall]
pub(super) fn near_empty_list(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    count: Ptr,
    list: Ptr,
) -> i32 {
    match near_ready(st) {
        Err(e) => e,
        Ok(()) if count.is_null() || list.is_null() => NEAR_ERROR_INVALID_ARGUMENT,
        Ok(()) => {
            ctx.write_u32(count.addr(), 0);
            0
        }
    }
}

/// The calls on a DISCOVERED gift: its info, sender and status, its image and received data
/// (open / read / close), and converting a launch parameter into one. No gift has been
/// discovered, so every id names nothing ("the specified gift does not exist"); outputs are
/// left alone.
pub(super) fn near_no_such_gift(ctx: &mut GuestCtx, st: &mut VitaState) {
    let r = near_ready(st).err().unwrap_or(NEAR_ERROR_RETRIEVE_FAILED);
    ctx.ret(r as u32);
}

/// int sceNearConvertDiscoveredGiftParam(const void *param, SceUInt32 *out)
#[hostcall]
pub(super) fn near_convert_discovered_gift_param(st: &mut VitaState, param: Ptr, out: Ptr) -> i32 {
    match near_ready(st) {
        Err(e) => e,
        Ok(()) if param.is_null() || out.is_null() => NEAR_ERROR_INVALID_ARGUMENT,
        Ok(()) => NEAR_ERROR_RETRIEVE_FAILED,
    }
}

/// int sceNearDeleteDiscoveredGift(SceUInt32 giftId)
#[hostcall]
pub(super) fn near_delete_discovered_gift(st: &mut VitaState, _id: u32) -> i32 {
    near_ready(st).err().unwrap_or(NEAR_ERROR_DELETION_FAILED)
}

/// int sceNearFinalizeAndLaunchNearApp(int mode, SceSize argSize, const void *arg)
/// int sceNearLaunchNearAppForUpdate(void)
///
/// Finalise the utility and ask the system to switch to the near app. There is no near app to
/// switch to, so the game keeps running, with the utility finalised as it asked.
pub(super) fn near_finalize_and_launch(ctx: &mut GuestCtx, st: &mut VitaState) {
    let r = near_finalize_in(st);
    ctx.ret(r as u32);
}

/// The state this module keeps on [`VitaState`].
#[derive(Default)]
pub struct NpState {
    lookup: Community,
    word_filter: Community,
    near: Near,
    /// Where the PS Store icon overlay was asked for, while shown.
    store_icon: Option<i32>,
    /// `sceNpBasicRegisterJoinGameAckHandler`'s `(handler, userdata)`, while registered.
    join_game_ack_handler: Option<(u32, u32)>,
    /// `sceNpMatching2Init`'s pool size, while initialised.
    matching2_pool: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::{SliceMemory, VFP_ARG_COUNT};
    use crate::world::DeterministicWorld;
    use vitaslop_transpiler::abi::REG_COUNT;

    type Entry = fn(&mut GuestCtx, &mut VitaState);

    const STACK: u32 = 0x700;

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
                bytes: vec![0u8; 0x800],
                st: VitaState::new(0, 0x800, Box::new(DeterministicWorld::default())),
            }
        }

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

        fn npid(&mut self, at: u32, name: &str, opt: [u8; 4], tag: [u8; 4]) -> u32 {
            let a = at as usize;
            self.bytes[a..a + NPID_SIZE].fill(0);
            self.bytes[a..a + name.len()].copy_from_slice(name.as_bytes());
            self.bytes[a + NPID_OPT..a + NPID_OPT + 4].copy_from_slice(&opt);
            self.bytes[a + NPID_PLATFORM..a + NPID_PLATFORM + 4].copy_from_slice(&tag);
            at
        }

        fn word(&self, at: u32) -> u32 {
            u32::from_le_bytes(self.bytes[at as usize..at as usize + 4].try_into().unwrap())
        }
    }

    /// Same name, account bytes and platform: equal. The name is case-sensitive and the account
    /// bytes count; an untagged pair matches, a tagged pair must agree.
    #[test]
    fn np_ids_compare_by_name_account_and_platform() {
        let mut r = Run::new();
        let a = r.npid(0x100, "player_one", [1, 2, 3, 4], [0; 4]);
        let b = r.npid(0x140, "player_one", [1, 2, 3, 4], [0; 4]);
        assert_eq!(r.call(cmp_np_id, &[a, b]), 0);
        let c = r.npid(0x180, "Player_one", [1, 2, 3, 4], [0; 4]);
        assert_eq!(r.call(cmp_np_id, &[a, c]), NP_UTIL_ERROR_NOT_MATCH);
        let d = r.npid(0x1C0, "player_one", [9, 2, 3, 4], [0; 4]);
        assert_eq!(r.call(cmp_np_id, &[a, d]), NP_UTIL_ERROR_NOT_MATCH);
        let e = r.npid(0x200, "player_one", [1, 2, 3, 4], *b"psp2");
        let f = r.npid(0x240, "player_one", [1, 2, 3, 4], *b"psp2");
        let g = r.npid(0x280, "player_one", [1, 2, 3, 4], *b"ps3\0");
        assert_eq!(r.call(cmp_np_id, &[e, f]), 0);
        assert_eq!(r.call(cmp_np_id, &[e, g]), NP_UTIL_ERROR_NOT_MATCH);
        assert_eq!(r.call(cmp_np_id, &[a, 0]), NP_UTIL_ERROR_INVALID_ARGUMENT);
        // A full 16-character name with no terminator inside the field.
        let h = r.npid(0x2C0, "abcdefghijklmnop", [0; 4], [0; 4]);
        let i = r.npid(0x300, "abcdefghijklmnop", [0; 4], [0; 4]);
        assert_eq!(r.call(cmp_np_id, &[h, i]), 0);
    }

    /// The platform tag is written at +0x18 and nothing else changes; an unknown type is refused.
    #[test]
    fn set_platform_type_writes_the_tag() {
        let mut r = Run::new();
        let a = r.npid(0x100, "x", [7; 4], [0; 4]);
        let before = r.bytes[0x100..0x100 + NPID_SIZE].to_vec();
        assert_eq!(r.call(set_platform_type, &[a, 2]), 0);
        assert_eq!(&r.bytes[0x118..0x11C], b"psp2");
        assert_eq!(r.bytes[0x100..0x118], before[..0x18]);
        assert_eq!(r.bytes[0x11C..0x124], before[0x1C..]);
        assert_eq!(r.call(set_platform_type, &[a, 0]), 0);
        assert_eq!(&r.bytes[0x118..0x11C], &[0; 4]);
        assert_eq!(
            r.call(set_platform_type, &[a, 4]),
            NP_ERROR_UNKNOWN_PLATFORM_TYPE
        );
        assert_eq!(
            r.call(set_platform_type, &[0, 2]),
            NP_ERROR_INVALID_ARGUMENT
        );
    }

    /// The object model: init once; contexts and requests from 1; unknown ids refused; a request
    /// carries one operation, which finishes NO_LOGIN and is then polled; abort makes it ABORTED;
    /// deleting a context deletes its requests; term clears everything.
    #[test]
    fn community_contexts_and_requests_are_real() {
        let mut r = Run::new();
        fn init(c: &mut GuestCtx, s: &mut VitaState) {
            community_init(c, s, Service::Lookup)
        }
        fn term(c: &mut GuestCtx, s: &mut VitaState) {
            community_term(c, s, Service::Lookup)
        }
        fn mk_ctx(c: &mut GuestCtx, s: &mut VitaState) {
            community_create_ctx(c, s, Service::Lookup)
        }
        fn del_ctx(c: &mut GuestCtx, s: &mut VitaState) {
            community_delete_ctx(c, s, Service::Lookup)
        }
        fn mk_req(c: &mut GuestCtx, s: &mut VitaState) {
            community_create_req(c, s, Service::Lookup)
        }
        fn abort(c: &mut GuestCtx, s: &mut VitaState) {
            community_abort_req(c, s, Service::Lookup)
        }
        fn sync(c: &mut GuestCtx, s: &mut VitaState) {
            community_sync_op(c, s, Service::Lookup)
        }
        fn start(c: &mut GuestCtx, s: &mut VitaState) {
            community_async_op(c, s, Service::Lookup)
        }
        fn poll(c: &mut GuestCtx, s: &mut VitaState) {
            community_completion(c, s, Service::Lookup)
        }
        fn wf_mk_ctx(c: &mut GuestCtx, s: &mut VitaState) {
            community_create_ctx(c, s, Service::WordFilter)
        }

        assert_eq!(
            r.call(mk_ctx, &[0x10, 0x20]),
            COMMUNITY_ERROR_NOT_INITIALIZED
        );
        assert_eq!(r.call(init, &[1, 0x1000_0100, 0, 0]), 0);
        assert_eq!(
            r.call(init, &[1, 0x1000_0100, 0, 0]),
            COMMUNITY_ERROR_ALREADY_INITIALIZED
        );
        // WordFilter keeps its own tables.
        assert_eq!(
            r.call(wf_mk_ctx, &[0x10, 0x20]),
            COMMUNITY_ERROR_NOT_INITIALIZED
        );
        let c1 = r.call(mk_ctx, &[0x10, 0x20]);
        assert_eq!(c1, 1);
        assert_eq!(
            r.call(mk_ctx, &[0, 0x20]),
            COMMUNITY_ERROR_INSUFFICIENT_ARGUMENT
        );
        let q1 = r.call(mk_req, &[c1 as u32]);
        assert_eq!(q1, 1);
        assert_eq!(r.call(mk_req, &[99]), COMMUNITY_ERROR_INVALID_ID);
        // Sync: refused with NO_LOGIN; the request is then spent.
        assert_eq!(
            r.call(sync, &[q1 as u32, 0x100, 0x140, 0]),
            COMMUNITY_ERROR_NO_LOGIN
        );
        assert_eq!(
            r.call(sync, &[q1 as u32, 0x100, 0x140, 0]),
            COMMUNITY_ERROR_INVALID_TYPE
        );
        // Async: starts, then the poll reports NO_LOGIN as the finished result.
        let q2 = r.call(mk_req, &[c1 as u32]) as u32;
        assert_eq!(r.call(poll, &[q2, 0x200]), COMMUNITY_ERROR_INVALID_ID);
        assert_eq!(r.call(start, &[q2, 0x100, 0x140, 0]), 0);
        assert_eq!(r.call(poll, &[q2, 0x200]), 0);
        assert_eq!(r.word(0x200) as i32, COMMUNITY_ERROR_NO_LOGIN);
        // Abort: the result is ABORTED from then on, and a new operation on it is refused.
        let q3 = r.call(mk_req, &[c1 as u32]) as u32;
        assert_eq!(r.call(abort, &[q3]), 0);
        assert_eq!(r.call(start, &[q3, 0, 0, 0]), COMMUNITY_ERROR_ABORTED);
        assert_eq!(r.call(poll, &[q3, 0x200]), 0);
        assert_eq!(r.word(0x200) as i32, COMMUNITY_ERROR_ABORTED);
        // Deleting the context takes its requests with it.
        assert_eq!(r.call(del_ctx, &[c1 as u32]), 0);
        assert_eq!(r.call(poll, &[q2, 0x200]), COMMUNITY_ERROR_INVALID_ID);
        assert_eq!(r.call(del_ctx, &[c1 as u32]), COMMUNITY_ERROR_INVALID_ID);
        // Ids are not reused.
        assert_eq!(r.call(mk_ctx, &[0x10, 0x20]), 2);
        assert_eq!(r.call(term, &[]), 0);
        assert_eq!(r.call(term, &[]), COMMUNITY_ERROR_NOT_INITIALIZED);
    }

    /// At most 32 contexts at once.
    #[test]
    fn community_limits_title_contexts() {
        let mut r = Run::new();
        fn init(c: &mut GuestCtx, s: &mut VitaState) {
            community_init(c, s, Service::WordFilter)
        }
        fn mk_ctx(c: &mut GuestCtx, s: &mut VitaState) {
            community_create_ctx(c, s, Service::WordFilter)
        }
        r.call(init, &[]);
        for i in 1..=32 {
            assert_eq!(r.call(mk_ctx, &[0x10, 0]), i);
        }
        assert_eq!(r.call(mk_ctx, &[0x10, 0]), COMMUNITY_ERROR_TOO_MANY_OBJECTS);
    }

    /// Near: initialise once; nothing discovered or near; a set gift comes back from GetGift
    /// (each blob up to the caller's capacity) and is deleted by its own id only.
    #[test]
    fn near_keeps_this_titles_gift_and_finds_nothing() {
        let mut r = Run::new();
        // work area { base, size }
        r.bytes[0x20..0x24].copy_from_slice(&0x400u32.to_le_bytes());
        r.bytes[0x24..0x28].copy_from_slice(&0x4_0000u32.to_le_bytes());
        assert_eq!(r.call(near_refresh, &[0x10]), NEAR_ERROR_NOT_INITIALIZED);
        assert_eq!(r.call(near_initialize, &[0x10, 0x20, 0]), 0);
        assert_eq!(
            r.call(near_initialize, &[0x10, 0x20, 0]),
            NEAR_ERROR_ALREADY_INITIALIZED
        );
        assert_eq!(r.call(near_empty_list, &[0x40, 0x44]), 0);
        assert_eq!(r.word(0x40), 0);
        assert_eq!(
            r.call(near_no_such_gift, &[5, 0x100]),
            NEAR_ERROR_RETRIEVE_FAILED
        );
        assert_eq!(
            r.call(near_delete_discovered_gift, &[5]),
            NEAR_ERROR_DELETION_FAILED
        );
        // No gift set yet.
        assert_eq!(
            r.call(
                near_get_gift,
                &[0x40, 0x100, 0x44, 0x300, 0x48, 0x320, 0x4C, 0x340]
            ),
            0
        );
        // Set one: info at 0x100 (0x1A0 bytes of 0xAB), image "IMG!", data "DATA-BLOB".
        r.bytes[0x100..0x2A0].fill(0xAB);
        r.bytes[0x2A0..0x2A4].copy_from_slice(b"IMG!");
        r.bytes[0x2B0..0x2B9].copy_from_slice(b"DATA-BLOB");
        assert_eq!(
            r.call(near_set_gift, &[77, 0x100, 4, 0x2A0, 9, 0x2B0, 3, 0]),
            0
        );
        // Read it back into fresh buffers; data capacity 4 truncates.
        r.bytes[0x100..0x2A0].fill(0);
        r.bytes[0x44..0x48].copy_from_slice(&16u32.to_le_bytes());
        r.bytes[0x48..0x4C].copy_from_slice(&4u32.to_le_bytes());
        assert_eq!(
            r.call(
                near_get_gift,
                &[0x40, 0x100, 0x44, 0x300, 0x48, 0x320, 0x4C, 0x340]
            ),
            1
        );
        assert_eq!(r.word(0x40), 77);
        assert!(r.bytes[0x100..0x2A0].iter().all(|&b| b == 0xAB));
        assert_eq!(r.word(0x44), 4);
        assert_eq!(&r.bytes[0x300..0x304], b"IMG!");
        assert_eq!(r.word(0x48), 4);
        assert_eq!(&r.bytes[0x320..0x324], b"DATA");
        assert_eq!(r.word(0x4C), 3);
        assert_eq!(r.call(near_delete_gift, &[78]), NEAR_ERROR_DELETION_FAILED);
        assert_eq!(r.call(near_delete_gift, &[77]), 0);
        assert_eq!(
            r.call(
                near_get_gift,
                &[0x40, 0x100, 0x44, 0x300, 0x48, 0x320, 0x4C, 0x340]
            ),
            0
        );
        // Launching the near app finalises; there is no app, so the game carries on.
        assert_eq!(r.call(near_finalize_and_launch, &[2, 4, 0x40]), 0);
        assert_eq!(r.call(near_refresh, &[0x10]), NEAR_ERROR_NOT_INITIALIZED);
    }

    /// The store icon is shown at one of three positions and hidden again; the empty NpBasic
    /// cache reports no blocked users and no friend presence.
    #[test]
    fn store_icon_and_basic_cache() {
        let mut r = Run::new();
        assert_eq!(r.call(commerce2_show_ps_store_icon, &[2]), 0);
        assert_eq!(r.st.np.store_icon, Some(2));
        assert_eq!(
            r.call(commerce2_show_ps_store_icon, &[3]),
            COMMERCE2_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(r.call(commerce2_hide_ps_store_icon, &[]), 0);
        assert_eq!(r.st.np.store_icon, None);
        assert_eq!(
            r.call(commerce2_abort_req, &[1]),
            COMMERCE2_ERROR_REQ_NOT_FOUND
        );
        r.bytes[0x40..0x44].copy_from_slice(&5u32.to_le_bytes());
        assert_eq!(r.call(basic_get_block_list_entry_count, &[0x40]), 0);
        assert_eq!(r.word(0x40), 0);
        assert_eq!(
            r.call(basic_get_block_list_entries, &[0, 0x100, 10, 0x40]),
            0
        );
        assert_eq!(
            r.call(basic_get_game_presence_of_friend, &[0x100, 0x200]),
            BASIC_ERROR_SIGNED_OUT
        );
    }

    /// The join-game handler registers locally (a null one refused) and unregisters with or
    /// without one in place.
    #[test]
    fn join_game_ack_handler_registers_locally() {
        let mut r = Run::new();
        assert_eq!(
            r.call(basic_register_join_game_ack_handler, &[0, 0]),
            BASIC_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(r.call(basic_register_join_game_ack_handler, &[0x81_0001, 7]), 0);
        assert_eq!(r.st.np.join_game_ack_handler, Some((0x81_0001, 7)));
        assert_eq!(r.call(basic_unregister_join_game_ack_handler, &[]), 0);
        assert_eq!(r.st.np.join_game_ack_handler, None);
        assert_eq!(r.call(basic_unregister_join_game_ack_handler, &[]), 0);
    }

    /// Matching2's pool reads back from `GetMemoryInfo` with nothing used, the local net info
    /// is signaling's NETINFO_NOT_AVAILABLE, and both refuse before `Init`.
    #[test]
    fn matching2_pool_and_local_net_info() {
        let mut r = Run::new();
        assert_eq!(
            r.call(matching2_get_memory_info, &[0x100]),
            MATCHING2_ERROR_NOT_INITIALIZED
        );
        assert_eq!(
            r.call(matching2_signaling_get_local_net_info, &[0x100]),
            MATCHING2_SIGNALING_ERROR_NOT_INITIALIZED
        );
        assert_eq!(r.call(matching2_init, &[0x2_0000, 100, 0, 0x4000]), 0);
        assert_eq!(
            r.call(matching2_init, &[0x2_0000, 100, 0, 0x4000]),
            MATCHING2_ERROR_ALREADY_INITIALIZED
        );
        assert_eq!(r.call(matching2_get_memory_info, &[0x100]), 0);
        assert_eq!(
            [r.word(0x100), r.word(0x104), r.word(0x108)],
            [0x2_0000, 0, 0]
        );
        assert_eq!(
            r.call(matching2_get_memory_info, &[0]),
            MATCHING2_ERROR_INVALID_ARGUMENT
        );
        assert_eq!(
            r.call(matching2_signaling_get_local_net_info, &[0x100]),
            MATCHING2_SIGNALING_ERROR_NETINFO_NOT_AVAILABLE
        );
        assert_eq!(r.call(matching2_term, &[]), 0);
        assert_eq!(
            r.call(matching2_get_memory_info, &[0x100]),
            MATCHING2_ERROR_NOT_INITIALIZED
        );
    }
}
