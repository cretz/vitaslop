//! Thread-local storage: each thread's `__thread` block (the `TPIDRURO` thread pointer) and
//! its `sceKernelGetTLSAddr` key-slot array.
//!
//! [`ThreadLocalStorage`] owns the template, the blocks, which blocks have had their `.tdata`
//! seeded, and the slot arrays. The caller resolves WHICH thread a request is for - a fiber
//! shares its runner's storage, which is `VitaState::logical_thread`'s question - and hands
//! in the allocator. It needs nothing else from the host.

use super::TLS_KEY_MAX;

/// Per-thread storage. See the module docs.
#[derive(Default)]
pub(super) struct ThreadLocalStorage {
    /// The main executable's TLS template `(init image address, initialized byte
    /// count, full block size)`, from [`crate::link::LinkedProgram::tls_template`].
    /// Each thread gets a private block of `memsz` bytes; the compiler reaches
    /// `__thread` variables at `thread_pointer + offset`. `(0, 0, 0)` = no TLS.
    template: (u32, u32, u32),
    /// Per-thread thread-pointer block base, keyed by thread id. Several threads can share a
    /// block - a fiber runs on its runner's.
    bases: Vec<(i32, u32)>,
    /// The blocks whose `.tdata` head has been seeded from the template - see
    /// [`ThreadLocalStorage::block_to_instantiate`]. Only a block's first instantiation may
    /// initialise it.
    seeded: Vec<u32>,
    /// Per-thread key-slot arrays behind `sceKernelGetTLSAddr`: `(thread, base)`, where key
    /// `k` is the pointer slot at `base + 4*k`. ONE zero-initialized array per thread, not a
    /// block per key: a caller may treat consecutive keys as one struct (a bundled libc keeps
    /// a `{array, count, generation}` triple in three adjacent keys), so key `k+1` must be the
    /// word after key `k`.
    slots: Vec<(i32, u32)>,
}

impl ThreadLocalStorage {
    pub(super) fn set_template(&mut self, template: (u32, u32, u32)) {
        self.template = template;
    }

    /// `(source address, initialized byte count)` of the template's `.tdata` image.
    pub(super) fn init_image(&self) -> (u32, u32) {
        (self.template.0, self.template.1)
    }

    /// Thread `thid`'s block, allocated through `galloc(size, align)` on first request; 0 when
    /// the title has no TLS.
    pub(super) fn block(&mut self, thid: i32, galloc: impl FnOnce(u32, u32) -> u32) -> u32 {
        let (_, _, memsz) = self.template;
        if memsz == 0 {
            return 0;
        }
        if let Some(&(_, base)) = self.bases.iter().find(|&&(t, _)| t == thid) {
            return base;
        }
        // 8-byte aligned so a `__thread` double or 64-bit value in the block is aligned.
        let base = galloc(memsz, 8);
        self.bases.push((thid, base));
        base
    }

    /// Bind thread `thid` to an existing block - a fiber's backing thread to its runner's.
    pub(super) fn bind(&mut self, thid: i32, base: u32) {
        self.bases.push((thid, base));
    }

    /// Forget thread `thid`'s binding; the block itself may still be another thread's.
    pub(super) fn forget_thread(&mut self, thid: i32) {
        self.bases.retain(|&(t, _)| t != thid);
    }

    /// `(thread pointer, init image source, bytes to copy)` for instantiating thread `thid`:
    /// the copy length is the template's `.tdata` size the FIRST time a block is handed out
    /// and 0 every time after. See `VitaState::tls_block_to_instantiate`.
    pub(super) fn block_to_instantiate(
        &mut self,
        thid: i32,
        galloc: impl FnOnce(u32, u32) -> u32,
    ) -> (u32, u32, u32) {
        let base = self.block(thid, galloc);
        let (src, len) = self.init_image();
        if base == 0 || self.seeded.contains(&base) {
            return (base, src, 0);
        }
        self.seeded.push(base);
        (base, src, len)
    }

    /// The address of slot `key` in `thread`'s slot array, allocating the array (zeroed)
    /// through `galloc(size, align)` on the thread's first query. `None` for a key past
    /// [`TLS_KEY_MAX`], or when the arena cannot spare the array.
    pub(super) fn slot_addr(
        &mut self,
        thread: i32,
        key: u32,
        galloc: impl FnOnce(u32, u32) -> u32,
    ) -> Option<u32> {
        if key > TLS_KEY_MAX {
            return None;
        }
        let base = match self.slots.iter().find(|&&(t, _)| t == thread) {
            Some(&(_, base)) => base,
            None => {
                let base = galloc((TLS_KEY_MAX + 1) * 4, 4);
                if base == 0 {
                    return None; // out of guest memory; the allocator has already said so
                }
                self.slots.push((thread, base));
                base
            }
        };
        Some(base + key * 4)
    }
}
