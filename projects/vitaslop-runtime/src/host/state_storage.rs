use super::*;

impl VitaState {
    /// Delete a file from the virtual filesystem, reporting whether it was there.
    /// Used by `sceAppUtilSaveDataDataRemove`, whose whole job is that the deletion is
    /// real: a title that removes a save slot and re-reads it must see it gone.
    pub fn remove_file(&mut self, path: &str) -> bool {
        let key = vfs_key(path);
        self.fs.originals.remove(&key);
        let gone = self.fs.files.remove(&key).is_some();
        // A DELETION is a change to the save, and it is the one that suffers most from
        // being missed: an unwritten write is progress the guest can make again, while an
        // unwritten delete leaves the old container on disk and the file the title just
        // removed comes back at the next reload. `sceAppUtilSaveDataDataRemove` is the
        // guest path here. Gated on the mount like every other site.
        if gone {
            self.fs.touch(&key);
        }
        gone
    }

    /// Total bytes the virtual filesystem holds under a mount (e.g. `savedata0:`),
    /// which is what a quota query has to report as USED.
    pub fn mount_used_bytes(&self, mount: &str) -> usize {
        let prefix = vfs_key(mount.trim_end_matches('/'));
        self.fs
            .files
            .iter()
            .filter(|(k, _)| k.starts_with(&prefix))
            .map(|(_, v)| v.len())
            .sum()
    }

    /// A fresh index for the next `scePhotoExportFromData`, so successive exports land
    /// in distinct files instead of overwriting one another.
    pub fn next_photo_export_index(&mut self) -> u32 {
        let i = self.photo_exports;
        self.photo_exports += 1;
        i
    }

    /// Move the heap allocation cursor to `addr`. A multi-module linked title
    /// (see [`crate::link`]) fills far more than the default 1 MiB below the heap,
    /// so the host must set this above the whole image (`LinkedProgram::alloc_base`)
    /// before the run, or allocations would overwrite guest code.
    pub fn set_alloc_base(&mut self, addr: u32) {
        self.arena.set_cursor(addr);
        // The fallback SA bank is placed HERE, before a single guest instruction runs, and
        // never moves. That is what lets its address live in the host-mirror block: the
        // mirror's contract is that a slot cannot change while guest code is running, and a
        // pointer fixed before the run trivially satisfies it. Allocating it lazily at the
        // first `sceGxmSetUniformDataF` would put a host call inside that contract for no
        // gain. See [`Self::sa_bank`].
        self.ensure_sa_bank();
    }

    /// Place the fallback SA bank if it is not placed yet, and return it.
    ///
    /// Called from [`Self::set_alloc_base`], which every real host calls before the guest
    /// runs - so on a real run the bank is a constant from before frame zero, which is what
    /// the mirror slot's contract wants. The lazy arm is for a harness that stands a
    /// `VitaState` up without an arena base and then uses it (the conformance cube does);
    /// there the slot goes 0 -> address once, and a resume that read the stale 0 simply
    /// ran the handler, which is this same code.
    pub(super) fn ensure_sa_bank(&mut self) -> u32 {
        if self.sa_bank == 0 {
            self.sa_bank = self.galloc(SA_BANK_BYTES, 16);
        }
        self.sa_bank
    }

    /// Guest address of the fallback SA bank, or 0 if the arena could not place one.
    ///
    /// Read by [`crate::vita::mirror::snapshot`] into the slot the inlined
    /// `sceGxmSetUniformDataF` reads it from.
    pub fn sa_bank(&self) -> u32 {
        self.sa_bank
    }

    /// Clear the bank for a new scene: zero the prefix anything wrote and reset the
    /// high-water mark.
    ///
    /// Only the prefix, because that is all a reader can see and all a later write can
    /// leave a hole in - `set_uniforms` fills the gap between the old high-water mark and
    /// its own start by RELYING on this being zero, which is the same thing the `Vec`
    /// this replaced got from `resize(end, 0.0)`.
    pub(super) fn clear_sa_bank(&mut self, ctx: &mut GuestCtx) {
        let bank = self.sa_bank;
        if bank == 0 {
            return;
        }
        let len = ctx.read_u32(bank);
        for i in 0..len.min(MAX_DEFAULT_UNIFORM_REGS) {
            ctx.write_u32(bank + SA_BANK_DATA + i * 4, 0);
        }
        ctx.write_u32(bank, 0);
    }

    /// The floats the bank holds, up to its high-water mark - what a draw with no default
    /// uniform buffer bound reads as its SA bank.
    pub(super) fn sa_bank_floats(&self, ctx: &GuestCtx) -> Vec<f32> {
        let bank = self.sa_bank;
        if bank == 0 {
            return Vec::new();
        }
        let len = ctx.read_u32(bank).min(MAX_DEFAULT_UNIFORM_REGS) as usize;
        ctx.read_f32s(bank + SA_BANK_DATA, len)
    }

    /// The same bank as BYTES - see `record_draw`'s vertex bank read for why both forms
    /// exist and why neither is derived from the other.
    pub(super) fn sa_bank_bytes(&self, ctx: &GuestCtx) -> Arc<[u8]> {
        let bank = self.sa_bank;
        if bank == 0 {
            return Arc::from(&[][..]);
        }
        let len = ctx.read_u32(bank).min(MAX_DEFAULT_UNIFORM_REGS) as usize;
        ctx.read_bytes_arc(bank + SA_BANK_DATA, len * 4)
    }

    // --- SceIoFilemgr virtual filesystem ---

    /// Put a whole file into the virtual filesystem at once. The guest can then
    /// `sceIoOpen`/`sceIoRead` it.
    ///
    /// Two callers, and the second is why this marks the save dirty. The first is the
    /// LOADER preloading a title's own read-only data before a run - `app0:`, which
    /// [`is_persisted_mount`] rejects, so `touch` is a no-op there. The second is a
    /// GUEST SAVE: `sceAppUtilSaveDataDataSave` writes a whole file per entry and lands
    /// here, and that is the API a title's autosave uses rather than
    /// `sceIoOpen`/`Write`/`Close`. Without the `touch` it wrote into the file table
    /// while [`VitaState::game_data_dirty`] stayed false, so the container was never
    /// written and the run's progress was gone at the next reload - the file was there,
    /// the whole time, and nothing asked for it to be saved.
    pub fn add_file(&mut self, path: &str, bytes: Vec<u8>) {
        let key = vfs_key(path);
        self.fs.originals.insert(
            key.clone(),
            strip_app0(path).trim_start_matches('/').to_string(),
        );
        self.fs.files.insert(key.clone(), bytes);
        self.fs.touch(&key);
        // An archive arrives as an ordinary file, so this is where a resident one is
        // mounted. `mount_archives` ignores a key it has already seen, so re-adding a file
        // does not re-parse it. See [`crate::psarc`].
        if key.ends_with(".psarc") {
            self.fs.mount_archives(&[key]);
        }
    }

    /// Serve the guest's read-only files from `backing` instead of loading them.
    ///
    /// The browser installs OPFS here: a retail container is over a gigabyte and the
    /// emulator's wasm32 heap tops out at four, so holding a title's assets in memory is
    /// what put a browser run over the edge. Native leaves this unset and keeps every
    /// file resident, which is what an in-process host should do.
    ///
    /// Call before the guest runs. Anything already added with [`add_file`](Self::add_file)
    /// stays resident and wins over a backed key of the same name.
    pub fn set_file_backing(&mut self, backing: Box<dyn FileBacking>) {
        self.fs.set_backing(backing);
    }

    /// Install one add-on: its files, served by `backing` under their paths relative to the
    /// add-on's own root, appear at `addcont0:/<dir>/...`, and `sceAppUtilDrmOpen(dir)` then
    /// reports it entitled.
    ///
    /// `dir` is the add-on's directory name - the 16-character label at the end of its
    /// content id (`XX0000-ABCD12345_00-<dir>`), which is what a device names its folder
    /// under `ux0:addcont/<title id>/` and what a title lists `addcont0:` to find. Call
    /// before the guest runs, after the title's own backing.
    pub fn add_addcont(&mut self, dir: &str, backing: Box<dyn FileBacking>) {
        self.note_addcont(dir);
        self.fs.set_backing(Box::new(MountedBacking::new(
            &format!("addcont0:/{dir}"),
            backing,
        )));
    }

    /// Record the add-on `dir` as installed when its files are already served under
    /// `addcont0:/<dir>/` by a backing installed some other way (the browser serves a title
    /// and its add-ons from one storage reader).
    pub fn note_addcont(&mut self, dir: &str) {
        if !self.has_addcont(dir) {
            self.addcont.push(dir.to_string());
        }
    }

    /// Whether the add-on `dir` is installed - case-insensitively, as the device's FAT is.
    pub fn has_addcont(&self, dir: &str) -> bool {
        self.addcont.iter().any(|d| d.eq_ignore_ascii_case(dir))
    }

    /// Read back a file's current bytes (a write target after the run, for tests).
    ///
    /// Resident files only, by design: this exists to inspect what a run WROTE, and a
    /// write always makes its file resident. A borrowed slice cannot be served from a
    /// backing without materialising the file, and doing that silently on an inspection
    /// call would hide the very residency this seam exists to avoid.
    pub fn file_bytes(&self, path: &str) -> Option<&[u8]> {
        self.fs.files.get(&vfs_key(path)).map(|v| v.as_slice())
    }

    /// sceIoOpen: returns a new fd or a negative errno.
    pub fn io_open(&mut self, path: &str, flags: u32) -> i32 {
        // >>> `tty0:` IS THE CONSOLE. A newer newlib does not ask the kernel for its standard
        // descriptors (`sceKernelGetStdout`), it OPENS them: `sceIoOpen("tty0:", O_RDONLY)` for
        // stdin and `O_WRONLY` twice for stdout and stderr. Refused as a missing file, a C
        // program's `printf` had no descriptor to write to and printed nothing, silently. A
        // write open is the captured console stream; a read open is stdin, which holds nothing.
        // Each open is its own descriptor, as on the console: a C library that keeps a table
        // of them reads a small reserved number as an empty slot.
        if path.eq_ignore_ascii_case("tty0:") {
            return self.fs.open_console(flags);
        }
        let fd = self.fs.open(path, flags);
        tracing::trace!(target: "vitaslop::io", path, flags = format_args!("{flags:#x}"), fd, "open");
        io_journal::push(|| {
            format!(
                "f{} t{:#x} open {path:?} {flags:#x} -> {fd}",
                self.cur_frame, self.current
            )
        });
        fd
    }

    /// One [`io_journal`] line, prefixed with the frame and the calling thread - for the
    /// file-layer handlers outside this impl (the FIOS2 overlays).
    pub fn journal(&self, line: impl FnOnce() -> String) {
        io_journal::push(|| format!("f{} t{:#x} {}", self.cur_frame, self.current, line()));
    }

    /// The path behind `fd` for a journal line, or `?`.
    pub(super) fn journal_path(&self, fd: i32) -> String {
        self.fs
            .open
            .get(&fd)
            .map(|o| o.path.clone())
            .unwrap_or_else(|| "?".into())
    }

    /// sceIoRead: read up to `len` bytes; None on a bad/unreadable fd.
    pub fn io_read(&mut self, fd: i32, len: usize) -> Option<Vec<u8>> {
        let at = self.fs.open.get(&fd).map(|o| o.cursor as u64);
        let r = self.fs.read(fd, len);
        let path = self
            .fs
            .open
            .get(&fd)
            .map(|o| o.path.clone())
            .unwrap_or_default();
        tracing::trace!(target: "vitaslop::io", fd, len, path, got = r.as_ref().map(|d| d.len()).unwrap_or(0), "read");
        io_journal::push(|| {
            format!(
                "f{} t{:#x} read fd{fd} {path:?} @{} len {len} -> {}",
                self.cur_frame,
                self.current,
                at.map_or("?".into(), |a| a.to_string()),
                io_journal::result(&r)
            )
        });
        r
    }

    /// sceIoPread: positioned read at `offset` that leaves the cursor untouched.
    pub fn io_pread(&mut self, fd: i32, offset: u64, len: usize) -> Option<Vec<u8>> {
        let r = self.fs.pread(fd, offset, len);
        io_journal::push(|| {
            format!(
                "f{} t{:#x} pread fd{fd} {:?} @{offset} len {len} -> {}",
                self.cur_frame,
                self.current,
                self.journal_path(fd),
                io_journal::result(&r)
            )
        });
        // Traced like `read`, and with the first bytes, because THIS is the call a title's
        // own archive layer uses and it was the one read the log could not see. Two engines
        // making the identical pread and then taking different branches is a difference in
        // the BYTES, and nothing else in the engine could show that: the arguments match, the
        // call count matches, and the divergence appears several thousand calls later.
        if tracing::enabled!(target: "vitaslop::io", tracing::Level::TRACE) {
            let path = self
                .fs
                .open
                .get(&fd)
                .map(|o| o.path.clone())
                .unwrap_or_default();
            let head: Vec<String> = r
                .as_ref()
                .map(|d| d.iter().take(16).map(|b| format!("{b:02x}")).collect())
                .unwrap_or_default();
            tracing::trace!(
                target: "vitaslop::io",
                fd, offset, len, path,
                got = r.as_ref().map(|d| d.len()).unwrap_or(0),
                head = head.join(""),
                "pread"
            );
        }
        r
    }

    /// sceIoPwrite: positioned write at `offset` that leaves the cursor untouched.
    pub fn io_pwrite(&mut self, fd: i32, offset: u64, bytes: &[u8]) -> Option<usize> {
        self.fs.pwrite(fd, offset, bytes)
    }

    /// sceIoWrite: fd 1/2 go to the captured console; other fds to the vfs.
    /// Returns the byte count written (always the full length for the console).
    pub fn io_write(&mut self, fd: i32, bytes: &[u8]) -> Option<usize> {
        match fd {
            FD_STDOUT => {
                self.capture.stdout.extend_from_slice(bytes);
                Some(bytes.len())
            }
            FD_STDERR => {
                self.capture.stderr.extend_from_slice(bytes);
                Some(bytes.len())
            }
            // A `tty0:` descriptor is the console, captured like fd 1.
            _ if self.fs.open.get(&fd).is_some_and(|o| o.console) => {
                self.capture.stdout.extend_from_slice(bytes);
                Some(bytes.len())
            }
            _ => self.fs.write(fd, bytes),
        }
    }

    /// sceIoLseek/sceIoLseek32: new absolute position or a negative errno.
    pub fn io_lseek(&mut self, fd: i32, offset: i64, whence: i32) -> i64 {
        let pos = self.fs.lseek(fd, offset, whence);
        io_journal::push(|| {
            format!(
                "f{} t{:#x} lseek fd{fd} {:?} {offset} whence {whence} -> {pos}",
                self.cur_frame,
                self.current,
                self.journal_path(fd)
            )
        });
        pos
    }

    /// sceIoClose: 0 or a negative errno.
    ///
    /// Egress ledger: if the descriptor being closed was a writable file under a
    /// persisted mount (`savedata0:`/`ux0:`), record a [`SaveWrite`](crate::capture::EgressKind::SaveWrite)
    /// with the path, final size, and an ASCII preview - the title-agnostic,
    /// human-readable "the game persisted state" signal a conformance check reads.
    /// Close is the natural commit point (the whole file is present by then).
    pub fn io_close(&mut self, fd: i32) -> i32 {
        if let Some(of) = self.fs.open.get(&fd)
            && of.writable
            && is_persisted_mount(&of.path)
        {
            let path = of.path.clone();
            if let Some(data) = self.fs.files.get(&path) {
                let ev = crate::capture::EgressEvent {
                    frame: self.cur_frame,
                    kind: crate::capture::EgressKind::SaveWrite {
                        path,
                        bytes: data.len(),
                        ascii: crate::capture::ascii_preview(data, 96),
                    },
                };
                self.capture.egress.push(ev);
            }
        }
        let path = self.journal_path(fd);
        let r = self.fs.close(fd);
        io_journal::push(|| {
            format!(
                "f{} t{:#x} close fd{fd} {path:?} -> {r}",
                self.cur_frame, self.current
            )
        });
        r
    }

    /// Record a [`SaveWrite`](crate::capture::EgressKind::SaveWrite) for a file the guest
    /// wrote WHOLE, rather than through `open`/`write`/`close`.
    ///
    /// `io_close` is the commit point for the descriptor path and cannot see this one:
    /// `sceAppUtilSaveDataDataSave` hands over a buffer per file and never opens anything,
    /// so a title whose autosave uses it wrote its progress with the ledger recording
    /// nothing - and the ledger is what a save check reads, so a conformance
    /// run for such a title could not show that it saved at all.
    ///
    /// Gated on the mount, like every other exportable-state site: a title rewriting one of
    /// its own shipped assets is not a save.
    pub fn record_save_write(&mut self, path: &str) {
        let key = vfs_key(path);
        if !is_persisted_mount(&key) {
            return;
        }
        let Some(data) = self.fs.files.get(&key) else {
            return;
        };
        let ev = crate::capture::EgressEvent {
            frame: self.cur_frame,
            kind: crate::capture::EgressKind::SaveWrite {
                path: key,
                bytes: data.len(),
                ascii: crate::capture::ascii_preview(data, 96),
            },
        };
        self.capture.egress.push(ev);
    }

    // --- the guest's own saved state ------------------------------------------

    /// Whether the guest has changed anything that would survive a restart, since the
    /// flag was last cleared. One bool over three stores, so a persistence layer can ask
    /// every frame and pay nothing on the frames where the answer is no.
    pub fn game_data_dirty(&self) -> bool {
        self.fs.dirty() || self.savedata.dirty() || self.trophies.dirty() || self.safe_memory_dirty
    }

    /// Copy `len` bytes of safe memory from `offset` (zeroes where the title never wrote).
    /// The caller has range-checked against [`SAFE_MEMORY_SIZE`].
    pub fn safe_memory_read(&self, offset: usize, len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        if let Some(src) = self.safe_memory.get(offset..offset + len) {
            out.copy_from_slice(src);
        }
        out
    }

    /// Write `bytes` into safe memory at `offset`, leaving the rest of it alone.
    pub fn safe_memory_write(&mut self, offset: usize, bytes: &[u8]) {
        if self.safe_memory.is_empty() {
            self.safe_memory = vec![0u8; SAFE_MEMORY_SIZE];
        }
        self.safe_memory[offset..offset + bytes.len()].copy_from_slice(bytes);
        self.safe_memory_dirty = true;
    }

    /// Clear the flag - call after the container has actually been written out, never
    /// before, or a flush that fails takes the record of the change with it.
    pub fn clear_game_data_dirty(&mut self) {
        self.fs.clear_dirty();
        self.savedata.clear_dirty();
        self.trophies.clear_dirty();
        self.safe_memory_dirty = false;
    }

    /// Everything this run has saved, as a portable container's worth of state.
    ///
    /// The installed title is NOT in here and cannot be: see
    /// [`FileTable::collect_game_data`] and [`is_persisted_mount`].
    pub fn game_data(&self) -> crate::gamedata::GameData {
        let mut data = self.fs.collect_game_data();
        data.slots = self.savedata.all();
        data.trophies = self.trophies.all_unlocked();
        data.safe_memory = self.safe_memory.clone();
        data
    }

    /// Put a previously exported container back, BEFORE the guest runs.
    ///
    /// Returns a one-line report of what landed. Applying this to a running guest would
    /// change files it may already hold descriptors on, so the caller is expected to do it
    /// at setup; nothing here enforces that, because there is no point at which this crate
    /// can tell.
    pub fn restore_game_data(&mut self, data: &crate::gamedata::GameData) -> String {
        let files = self.fs.restore_game_data(data);
        for (mount, id, param) in &data.slots {
            self.savedata.put(mount, *id, param.clone());
        }
        let trophies = self.trophies.restore_unlocked(&data.trophies);
        // A container from somewhere else may carry a different size; keep what fits.
        if !data.safe_memory.is_empty() {
            let n = data.safe_memory.len().min(SAFE_MEMORY_SIZE);
            self.safe_memory_write(0, &data.safe_memory[..n]);
        }
        // A restore is not a change TO be saved - what was just read in is already in the
        // container it came from - so the dirty flags the puts above set are cleared here.
        // Leaving them set would make the first tick of every run rewrite storage.
        self.clear_game_data_dirty();
        format!(
            "restored {files} file(s) ({} bytes), {} dir(s), {} slot(s), {trophies} trophy/ies",
            data.bytes(),
            data.dirs.len(),
            data.slots.len()
        )
    }

    // --- FIOS2 path overlays -------------------------------------------------

    /// Register an overlay, returning its id (see [`FiosOverlays`]).
    #[inline]
    pub fn fios_overlay_add(&mut self, overlay: FiosOverlay) -> i32 {
        self.fios_overlays.add(overlay)
    }

    /// The overlay with this id, if it is still registered.
    #[inline]
    pub fn fios_overlay(&self, id: i32) -> Option<&FiosOverlay> {
        self.fios_overlays.get(id)
    }

    /// Replace a registered overlay's configuration, keeping its id. Returns whether
    /// the id resolved.
    #[inline]
    pub fn fios_overlay_modify(&mut self, id: i32, new_value: FiosOverlay) -> bool {
        self.fios_overlays.modify(id, new_value)
    }

    /// Unregister an overlay. Returns whether the id resolved.
    #[inline]
    pub fn fios_overlay_remove(&mut self, id: i32) -> bool {
        self.fios_overlays.remove(id)
    }

    /// The ids of every overlay whose `order` falls in `[min_order, max_order]`, in
    /// application order.
    pub fn fios_overlay_ids(&self, min_order: u8, max_order: u8) -> Vec<i32> {
        self.fios_overlays
            .in_orders(min_order, max_order)
            .map(|o| o.id)
            .collect()
    }

    /// Whether the calling thread has switched overlay resolution off for itself.
    #[inline]
    pub fn fios_overlay_disabled(&self) -> bool {
        self.fios_overlays.disabled_for(self.current)
    }

    /// Switch overlay resolution on or off for the calling thread. Returns the
    /// previous setting, which is what a caller saves to restore.
    #[inline]
    pub fn fios_overlay_set_disabled(&mut self, disabled: bool) -> bool {
        self.fios_overlays.set_disabled_for(self.current, disabled)
    }

    /// Resolve `path` through the registered overlays whose `order` is in
    /// `[min_order, max_order]`, returning the path the filesystem should actually
    /// see.
    ///
    /// Each overlay whose `dst` is a path prefix of the current path can rewrite that
    /// prefix to its `src`. What decides whether it does is the overlay TYPE:
    ///
    /// - `OPAQUE` replaces unconditionally: `src` stands in for `dst` whether or not
    ///   anything is there, which is what makes it opaque - the original is hidden.
    /// - `TRANSLUCENT` and `WRITABLE` check `src` first and fall back to `dst`, so the
    ///   rewrite only takes when the file is really present in the overlay. (They
    ///   differ in where WRITES land, which is a property of the open, not of the
    ///   resolve.)
    /// - `NEWER` picks whichever copy has the later modification time. Nothing in
    ///   this filesystem carries a modification time unless a title set one with
    ///   `sceIoChstat`, and the type's own documented tie-break is "if both have the
    ///   same modification time, dst is used" - which is exactly the untimed case, so
    ///   an untimed pair resolves to `dst` unless only `src` exists.
    ///
    /// Overlays compose: a rewritten path is fed to the next overlay in order, so a
    /// chain of them behaves as a stack.
    pub fn fios_resolve(&self, path: &str, min_order: u8, max_order: u8) -> String {
        if self.fios_overlay_disabled() {
            return path.to_string();
        }
        let mut cur = path.to_string();
        for o in self.fios_overlays.in_orders(min_order, max_order) {
            let Some(rest) = strip_path_prefix(&cur, &o.dst) else {
                continue;
            };
            let candidate = format!("{}{rest}", o.src);
            let exists = |p: &str| self.io_size(p).is_some() || self.io_is_dir(p);
            cur = match o.kind {
                SCE_FIOS_OVERLAY_TYPE_OPAQUE => candidate,
                SCE_FIOS_OVERLAY_TYPE_TRANSLUCENT | SCE_FIOS_OVERLAY_TYPE_WRITABLE => {
                    if exists(&candidate) { candidate } else { cur }
                }
                SCE_FIOS_OVERLAY_TYPE_NEWER => {
                    if exists(&cur) {
                        cur
                    } else if exists(&candidate) {
                        candidate
                    } else {
                        cur
                    }
                }
                // An overlay type the console does not define. Leaving the path alone
                // would silently ignore a registered overlay, so say so once and move
                // on rather than pretending it applied.
                other => {
                    tracing::error!(
                        target: "vitaslop::err",
                        kind = other, id = o.id, dst = %o.dst, src = %o.src,
                        "FIOS2 overlay has an undefined type - it is being ignored, so \
                         paths under its dst are NOT being redirected"
                    );
                    cur
                }
            };
        }
        cur
    }

    /// int sceIoMkdir: create a directory (see [`FileTable::mkdir`]).
    pub fn io_mkdir(&mut self, path: &str) -> i32 {
        self.fs.mkdir(path)
    }

    /// int sceIoRmdir: remove an empty directory (see [`FileTable::rmdir`]).
    pub fn io_rmdir(&mut self, path: &str) -> i32 {
        self.fs.rmdir(path)
    }

    /// int sceIoRemove: delete a file (see [`FileTable::remove`]).
    pub fn io_remove(&mut self, path: &str) -> i32 {
        self.fs.remove(path)
    }

    /// int sceIoRename: move a file or directory subtree (see [`FileTable::rename`]).
    pub fn io_rename(&mut self, old: &str, new: &str) -> i32 {
        self.fs.rename(old, new)
    }

    /// int sceIoChstat: apply the selected status fields to a path.
    pub fn io_chstat(&mut self, path: &str, over: FileStatOverride) -> i32 {
        self.fs.chstat(path, over)
    }

    pub fn io_chstat_fd(&mut self, fd: i32, over: FileStatOverride) -> i32 {
        self.fs.chstat_fd(fd, over)
    }

    /// The chstat overrides recorded for a path, if any.
    pub fn io_stat_override(&self, path: &str) -> Option<&FileStatOverride> {
        self.fs.stat_override(path)
    }

    /// Whether a path names a directory (explicit or implied by its contents).
    pub fn io_is_dir(&self, path: &str) -> bool {
        self.fs.is_dir(&vfs_key(path))
    }

    /// Whether `fd` is a live DIRECTORY descriptor.
    pub fn io_dir_is_open(&self, fd: i32) -> bool {
        self.fs.open_dirs.contains_key(&fd)
    }

    /// The path a live directory descriptor was opened on.
    pub fn io_dir_path(&self, fd: i32) -> Option<String> {
        self.fs.open_dirs.get(&fd).map(|d| d.path.clone())
    }

    pub fn io_dopen(&mut self, path: &str) -> i32 {
        let fd = self.fs.dopen(path);
        tracing::trace!(target: "vitaslop::io", path, fd, "dopen");
        io_journal::push(|| {
            format!(
                "f{} t{:#x} dopen {path:?} -> {fd}",
                self.cur_frame, self.current
            )
        });
        fd
    }

    /// sceIoDread: the next entry of an open directory. `None` = bad descriptor,
    /// `Some(None)` = end of listing.
    pub fn io_dread(&mut self, fd: i32) -> Option<Option<DirEntry>> {
        let r = self.fs.dread(fd);
        let name = match &r {
            Some(Some(e)) => e.name.clone(),
            Some(None) => "<end>".into(),
            None => "<badfd>".into(),
        };
        tracing::trace!(target: "vitaslop::io", fd, name, "dread");
        r
    }

    /// sceIoDclose: 0 or a negative errno.
    pub fn io_dclose(&mut self, fd: i32) -> i32 {
        self.fs.dclose(fd)
    }

    /// File size for sceIoGetstat, or None if the path does not exist.
    pub fn io_size(&self, path: &str) -> Option<u64> {
        let r = self.fs.size_of(path);
        tracing::trace!(target: "vitaslop::io", path, size = r.unwrap_or(u64::MAX), found = r.is_some(), "getstat");
        io_journal::push(|| {
            format!(
                "f{} t{:#x} getstat {path:?} -> {r:?}",
                self.cur_frame, self.current
            )
        });
        r
    }

    /// The size of the file behind an open descriptor (for sceIoGetstatByFd).
    pub fn io_size_fd(&self, fd: i32) -> Option<u64> {
        // The console holds no bytes, but it is a valid descriptor to stat.
        let console = self.fs.open.get(&fd).is_some_and(|o| o.console);
        let r = if console { Some(0) } else { self.fs.size_of_fd(fd) };
        io_journal::push(|| {
            format!(
                "f{} t{:#x} getstat fd{fd} {:?} -> {r:?}",
                self.cur_frame,
                self.current,
                self.journal_path(fd)
            )
        });
        r
    }

    /// The whole contents of a guest file by path (for one-shot loads like a font
    /// file), or `None` if it does not exist.
    pub fn read_file(&self, path: &str) -> Option<Vec<u8>> {
        self.fs.read_all(path)
    }

    /// The entries directly under `path`, for a host call that must DISCOVER a name
    /// rather than being handed one - a system service resolving the running title's
    /// own data, where the console knows the name from the installed package and we
    /// only have the package's layout. Empty if the directory does not exist.
    pub fn list_dir(&mut self, path: &str) -> Vec<DirEntry> {
        let fd = self.fs.dopen(path);
        if fd < 0 {
            return Vec::new();
        }
        let mut out = Vec::new();
        while let Some(Some(e)) = self.fs.dread(fd) {
            out.push(e);
        }
        self.fs.dclose(fd);
        out
    }

    /// Whether a savedata slot exists (SceAppUtil metadata layer).
    pub fn savedata_slot_exists(&self, mount: &str, slot_id: u32) -> bool {
        self.savedata.contains(mount, slot_id)
    }

    /// Create or overwrite a savedata slot's param blob.
    pub fn savedata_slot_put(&mut self, mount: &str, slot_id: u32, param: Vec<u8>) {
        self.savedata.put(mount, slot_id, param);
    }

    /// The stored param blob for a savedata slot, cloned for the caller to write back
    /// into guest memory (`None` if the slot was never created).
    pub fn savedata_slot_get(&self, mount: &str, slot_id: u32) -> Option<Vec<u8>> {
        self.savedata.get(mount, slot_id).map(<[u8]>::to_vec)
    }

    /// Remove a savedata slot; returns whether it existed.
    pub fn savedata_slot_remove(&mut self, mount: &str, slot_id: u32) -> bool {
        self.savedata.remove(mount, slot_id)
    }
}
