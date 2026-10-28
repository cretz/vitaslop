//! The shader patcher's live programs: how many references each holds, and the vertex programs
//! it can hand back again because they were created with identical parameters.
//!
//! [`ProgramRefs`] owns both maps; the `VitaState` methods the GXM handlers call forward to it
//! from `state_kobjects.rs`. It needs nothing from the rest of the host.

use std::collections::{BTreeMap, HashMap};

/// The key a vertex program is shared under: its `SceGxmProgram*` plus the raw attribute and
/// stream arrays it was created with.
pub(super) type ShareKey = (u32, Vec<u8>);

/// Live shader-patcher programs. See the module docs.
#[derive(Default)]
pub(super) struct ProgramRefs {
    /// Live shader-patcher programs and how many references each holds, for
    /// `sceGxmShaderPatcherGet{Vertex,Fragment}ProgramRefCount`.
    ///
    /// >>> WHY A COUNT AND NOT A SET. On the console the patcher SHARES programs: two
    /// >>> `CreateVertexProgram` calls with identical parameters hand back the same program
    /// >>> with its reference count at two, and a title that releases once still has a live
    /// >>> program. This patcher shares vertex programs the same way (see `share`), so a count
    /// >>> is the only thing that can answer the getter truthfully.
    refs: BTreeMap<u32, u32>,
    /// Live vertex programs by their CREATION PARAMETERS, so an identical `CreateVertexProgram`
    /// hands back the SAME program with one more reference, as the console's patcher does. See
    /// `gxm::create_vertex_program`. The reverse map drops the entry at the last release.
    share: HashMap<ShareKey, u32>,
    share_key: HashMap<u32, ShareKey>,
}

impl ProgramRefs {
    pub(super) fn created(&mut self, handle: u32) {
        if handle != 0 {
            *self.refs.entry(handle).or_insert(0) += 1;
        }
    }

    pub(super) fn released(&mut self, handle: u32) -> bool {
        match self.refs.get_mut(&handle) {
            None => false,
            Some(n) => {
                *n -= 1;
                if *n == 0 {
                    self.refs.remove(&handle);
                    if let Some(key) = self.share_key.remove(&handle) {
                        self.share.remove(&key);
                    }
                }
                true
            }
        }
    }

    pub(super) fn share(&mut self, key: &ShareKey) -> Option<u32> {
        let handle = *self.share.get(key)?;
        let n = self.refs.get_mut(&handle)?;
        *n += 1;
        Some(handle)
    }

    pub(super) fn remember(&mut self, key: ShareKey, handle: u32) {
        if handle != 0 {
            self.share_key.insert(handle, key.clone());
            self.share.insert(key, handle);
        }
    }

    pub(super) fn count(&self, handle: u32) -> Option<u32> {
        self.refs.get(&handle).copied()
    }
}
