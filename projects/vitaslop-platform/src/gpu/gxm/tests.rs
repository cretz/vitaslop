//! The renderer's unit tests that span its files: GXM enum mappings, the fallback evidence,
//! the wasm clock rule, the texture budget and its accounting, the cache bounds, and the
//! content hash.

use super::*;

#[cfg(test)]
/// `std::time::Instant` must not be constructed anywhere in this file except inside
/// [`Stopwatch`], because it PANICS on `wasm32-unknown-unknown` and this file is compiled
/// for the browser.
///
/// # Why a source grep rather than a type
/// The panic is a RUNTIME one: `Instant::now()` compiles for wasm32 perfectly happily and
/// dies the moment it is called, so neither the wasm build nor the desktop tests can catch
/// it. It shipped exactly that way - a pipeline-build timer added to `build_gxp_pipeline`
/// killed the browser's run worker on the first frame it rendered anything, with a stack
/// reading `time not implemented on this platform`, while `cargo build --target
/// wasm32-unknown-unknown` was green. The doc comment on `wasm_now` had warned about this
/// for a whole session before that happened, which is the argument for a test over a
/// comment.
/// The guest's stencil enums reach the pipeline unchanged in meaning. A HUD's whole shape
/// is a `NEVER` quad whose FAIL op REPLACEs a bit and an `EQUAL` fill that tests it; a
/// misplaced func field or op table draws every masked quad whole.
#[cfg(test)]
mod stencil_mapping_tests {
    use super::{gxm_stencil_func, gxm_stencil_op};
    use wgpu::{CompareFunction as C, StencilOperation as S};

    #[test]
    fn the_stencil_func_is_bits_27_to_25_in_the_depth_order() {
        // `SceGxmStencilFunc` values as a title passes them (vitasdk `gxm.h`).
        let table = [
            (0x0000_0000u32, C::Never),
            (0x0200_0000, C::Less),
            (0x0400_0000, C::Equal),
            (0x0600_0000, C::LessEqual),
            (0x0800_0000, C::Greater),
            (0x0a00_0000, C::NotEqual),
            (0x0c00_0000, C::GreaterEqual),
            (0x0e00_0000, C::Always),
        ];
        for (raw, want) in table {
            assert_eq!(gxm_stencil_func(raw), want, "{raw:#010x}");
        }
    }

    #[test]
    fn the_stencil_ops_are_the_vitasdk_order() {
        let table = [
            (0u32, S::Keep),
            (1, S::Zero),
            (2, S::Replace),
            (3, S::IncrementClamp),
            (4, S::DecrementClamp),
            (5, S::Invert),
            (6, S::IncrementWrap),
            (7, S::DecrementWrap),
        ];
        for (raw, want) in table {
            assert_eq!(gxm_stencil_op(raw), want, "{raw}");
        }
    }
}

#[cfg(test)]
mod color_mask_tests {
    use super::gxm_color_mask;
    use wgpu::ColorWrites as W;

    /// `SceGxmColorMask` (vitasdk `gxm.h`): A is bit 0, then R, G, B. Mask 14 is the one a
    /// title's hair pass uses - every colour channel, the destination alpha kept.
    #[test]
    fn the_colour_mask_is_alpha_first_in_the_vitasdk_order() {
        assert_eq!(gxm_color_mask(1), W::ALPHA);
        assert_eq!(gxm_color_mask(2), W::RED);
        assert_eq!(gxm_color_mask(4), W::GREEN);
        assert_eq!(gxm_color_mask(8), W::BLUE);
        assert_eq!(gxm_color_mask(14), W::COLOR, "RGB with the destination alpha kept");
        assert_eq!(gxm_color_mask(15), W::ALL);
        assert_eq!(gxm_color_mask(0), W::empty());
    }
}

#[cfg(test)]
mod fallback_evidence_tests {
    /// The base64 in a fatal refusal is hand-rolled (this crate has no encoder and one
    /// dependency for one diagnostic is not a trade worth making), so it is checked against
    /// the RFC 4648 vectors - including every padding case, which is where a hand-rolled
    /// encoder goes wrong and where a wrong byte would silently corrupt the one copy of a
    /// shader nobody can reproduce on demand.
    #[test]
    fn the_evidence_base64_round_trips_the_standard_vectors() {
        let b64 = super::base64;
        assert_eq!(b64(b""), "");
        assert_eq!(b64(b"f"), "Zg==");
        assert_eq!(b64(b"fo"), "Zm8=");
        assert_eq!(b64(b"foo"), "Zm9v");
        assert_eq!(b64(b"foob"), "Zm9vYg==");
        assert_eq!(b64(b"fooba"), "Zm9vYmE=");
        assert_eq!(b64(b"foobar"), "Zm9vYmFy");
        // The two characters a 6-bit table gets wrong if the tail is mis-indexed, and the
        // high bytes a signed shift would mangle.
        assert_eq!(b64(&[0xff, 0xef, 0xfe]), "/+/+");
        assert_eq!(b64(&[0x00, 0x00, 0x00]), "AAAA");
    }

    /// The payload is armed per pair and consumed ONCE: a second refusal must not re-print
    /// kilobytes, and a refusal for a DIFFERENT pair must not print the armed one's bytes
    /// under the wrong key - which would send someone to fix the wrong shader.
    #[test]
    fn the_evidence_is_keyed_to_its_pair_and_spent_once() {
        let v: std::sync::Arc<[u8]> = std::sync::Arc::from(&b"vert"[..]);
        let f: std::sync::Arc<[u8]> = std::sync::Arc::from(&b"frag"[..]);
        super::arm_fallback_blobs(0xAAAA, v.clone(), f.clone());
        assert_eq!(super::blob_evidence(0xBBBB), "", "another pair gets nothing");
        super::arm_fallback_blobs(0xAAAA, v, f);
        let first = super::blob_evidence(0xAAAA);
        assert!(first.contains("dmVydA=="), "carries the vertex container: {first}");
        assert!(first.contains("ZnJhZw=="), "carries the fragment container: {first}");
        assert_eq!(super::blob_evidence(0xAAAA), "", "spent - a second refusal is quiet");
    }
}

#[cfg(test)]
mod wasm_clock_tests {
    /// The only lines allowed to name it: the `Stopwatch` field, its constructor, and prose.
    #[test]
    fn nothing_constructs_a_std_instant_outside_the_stopwatch() {
        // Spelled in halves so this test's OWN source does not match the needle - the
        // first version of it failed on itself, which is funny once and then is just a
        // broken test.
        let needle = concat!("std::time::", "Instant", "::now()");
        // EVERY file of the renderer, not one: the module is split across `gpu.rs` and the
        // files under `gpu/gxm/`, and a rule checked against one of them is a rule the rest
        // are free to break.
        let sources = [
            ("gpu.rs", include_str!("../../gpu.rs")),
            ("render.rs", include_str!("../render.rs")),
            ("gxm/mod.rs", include_str!("mod.rs")),
            ("gxm/buffers.rs", include_str!("buffers.rs")),
            ("gxm/build_pipeline.rs", include_str!("build_pipeline.rs")),
            ("gxm/caches.rs", include_str!("caches.rs")),
            ("gxm/clip.rs", include_str!("clip.rs")),
            ("gxm/coverage.rs", include_str!("coverage.rs")),
            ("gxm/encode.rs", include_str!("encode.rs")),
            ("gxm/fallback.rs", include_str!("fallback.rs")),
            ("gxm/gxp_types.rs", include_str!("gxp_types.rs")),
            ("gxm/inputs.rs", include_str!("inputs.rs")),
            ("gxm/live.rs", include_str!("live.rs")),
            ("gxm/live_bind.rs", include_str!("live_bind.rs")),
            ("gxm/live_prepare.rs", include_str!("live_prepare.rs")),
            ("gxm/pipeline_state.rs", include_str!("pipeline_state.rs")),
            ("gxm/prepare.rs", include_str!("prepare.rs")),
            ("gxm/renderer_chain.rs", include_str!("renderer_chain.rs")),
            ("gxm/renderer_misc.rs", include_str!("renderer_misc.rs")),
            ("gxm/renderer_new.rs", include_str!("renderer_new.rs")),
            ("gxm/renderer_pass.rs", include_str!("renderer_pass.rs")),
            ("gxm/renderer_resources.rs", include_str!("renderer_resources.rs")),
            ("gxm/renderer_rtt.rs", include_str!("renderer_rtt.rs")),
            ("gxm/renderer_warm.rs", include_str!("renderer_warm.rs")),
            ("gxm/repack.rs", include_str!("repack.rs")),
            ("gxm/reports.rs", include_str!("reports.rs")),
            ("gxm/resident.rs", include_str!("resident.rs")),
            ("gxm/rtt.rs", include_str!("rtt.rs")),
            ("gxm/tests.rs", include_str!("tests.rs")),
            ("gxm/textures.rs", include_str!("textures.rs")),
            ("gxm/timestamps.rs", include_str!("timestamps.rs")),
            ("gxm/uniforms.rs", include_str!("uniforms.rs")),
            ("gxm/uploads.rs", include_str!("uploads.rs")),
            ("gxm/vertex.rs", include_str!("vertex.rs")),
        ];
        let offenders: Vec<(&str, usize, &str)> = sources
            .iter()
            .flat_map(|(file, src)| {
                src.lines()
                    .enumerate()
                    .filter(|(_, l)| l.contains(needle))
                    .filter(|(_, l)| !l.contains("Stopwatch { start:"))
                    // Prose, including the doc comment on this very test.
                    .filter(|(_, l)| !l.trim_start().starts_with("///"))
                    .filter(|(_, l)| !l.trim_start().starts_with("//"))
                    .map(move |(n, l)| (*file, n + 1, l.trim()))
            })
            .collect();
        assert!(
            offenders.is_empty(),
            "{needle} panics on wasm32 and this file runs in the browser - use `Stopwatch::start()`. Offending lines: {offenders:?}"
        );
    }
}

mod texture_budget_default_tests {
    /// The default budget is the CONSOLE's ceiling on resident game memory, and it is
    /// spelled as its three partitions so that it cannot quietly become a round number
    /// somebody liked. The previous value was fitted to one title's menu screen and cost a
    /// race 83% re-decodes on the target device; a figure with no derivation attached is how
    /// that happens again.
    #[test]
    fn the_default_is_the_consoles_game_partitions_and_nothing_else() {
        assert_eq!(
            super::GAME_RESIDENT_CEILING_MB,
            256 + 109 + 112,
            "ScePhyMemPartGame + the +109 MiB extension + ScePhyMemPartGameCdram"
        );
        assert_eq!(super::GAME_RESIDENT_CEILING_MB, 477);
    }

    /// The PRESSURE signal and the UPLOADER must price the budget identically.
    ///
    /// They did not. `texture_budget_pressure` carried its own copy of this arithmetic with
    /// `unwrap_or(256)` under a comment saying "the same budget the uploader enforces, read
    /// the same way, so the two cannot disagree about what tight means" - while the uploader
    /// defaulted to 477. Pressure therefore declared itself at 171 MiB where the uploader's
    /// own two-thirds threshold is 318, and on a no-BC adapter every screen re-encoded its BC
    /// textures to ETC2 - a block decode plus an alpha-carrying encode, the most expensive
    /// path there is - to save memory that was not short.
    ///
    /// The two now call one function. This test is what stops a second copy appearing: a
    /// duplicated default is invisible precisely because SETTING the knob moves both.
    /// Both tests below write the one global working set; run in parallel, one's store
    /// landed between the other's store and its assertion.
    #[cfg(test)]
    static WORKING_SET_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn the_pressure_signal_prices_the_budget_the_same_way_the_uploader_does() {
        let _g = WORKING_SET_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // 0 is the "no frame has finished" sentinel, which is pressure by design, so the
        // comparison is made at a working set that is unambiguously below any threshold.
        crate::gpu::LAST_WORKING_SET.store(1, std::sync::atomic::Ordering::Relaxed);
        let budget = super::tex_cache_budget_bytes();
        assert!(
            !crate::gpu::texture_budget_pressure(),
            "1 byte resident is not pressure against a {} MiB budget",
            budget / (1024 * 1024)
        );
        // ...and at two-thirds of that same budget it MUST be, or the two readers are using
        // different numbers again.
        crate::gpu::LAST_WORKING_SET.store(budget / 3 * 2 + 1, std::sync::atomic::Ordering::Relaxed);
        assert!(crate::gpu::texture_budget_pressure());
        crate::gpu::LAST_WORKING_SET.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// >>> THE RETENTION BOUND MUST NOT BE THE RE-ENCODE GATE.
    ///
    /// This is the whole reason `tex_retain_budget_bytes` exists as a second number. The
    /// view cache was bounded by `tex_cache_budget_bytes`, which is ALSO what
    /// `texture_budget_pressure` reads to decide whether a BC texture is re-encoded to ETC2 -
    /// a second lossy step over the guest's own compression. So the obvious way to stop the
    /// cache retaining 343 MB on a phone (lower the budget) would have bought that memory
    /// with PICTURE, silently, on the engine nobody can look at.
    ///
    /// The test states the invariant rather than the intent: retention is well under the
    /// budget, and pressure is unmoved by it.
    #[test]
    fn bounding_retention_does_not_tighten_the_re_encode_gate() {
        let _g = WORKING_SET_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let budget = super::tex_cache_budget_bytes();
        let retain = super::tex_retain_budget_bytes();
        assert!(
            retain <= budget,
            "a retention bound above the budget would make VITASLOP_TEX_CACHE_MB a no-op                  for retention: {retain} > {budget}"
        );
        // A working set the size of the RETENTION bound is not, on its own, budget pressure:
        // if it were, bounding retention would have started re-encoding textures that fit.
        crate::gpu::LAST_WORKING_SET.store(retain, std::sync::atomic::Ordering::Relaxed);
        assert!(
            !crate::gpu::texture_budget_pressure(),
            "a working set at the retention bound ({} MiB) must not read as pressure against                  the {} MiB texture budget - that is the quality trade this split exists to avoid",
            retain / (1024 * 1024),
            budget / (1024 * 1024),
        );
        crate::gpu::LAST_WORKING_SET.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod texture_budget_accounting_tests {
    use super::{texture_upload_bytes, BlockFormat, CompressedUpload, TexelSeam};

    /// [`texture_upload_bytes`] on the decode path - the device gate is what the other
    /// argument is, and it has its own test below.
    fn uploaded_texture_bytes(t: &super::GxmTexture) -> usize {
        texture_upload_bytes(t, false)
    }

    fn tex(width: u32, height: u32, faces: u32, texel: TexelSeam) -> super::GxmTexture {
        super::GxmTexture {
            // A budget fixture: no guest bytes to expand on the GPU.
            raw: None,
            key: 0,
            data_addr: 0,
            // Irrelevant to a MIP-accounting fixture: the flag only decides whether a
            // render target keeps its alias - see `GxmTexture::guest_bytes_unwritten`.
            guest_bytes_unwritten: false,
            row_bytes: 0,
            width,
            height,
            faces,
            rgba: crate::gpu::Texels::ready(Vec::new()),
            texel,
            // These tests are about the MIP ACCOUNTING - that the budget and the uploader
            // agree on what a chain costs - so the fixture has to be a texture that gets
            // one: several guest levels. A texture the guest gave ONE level gets no chain
            // (the device samples level 0 alone) - see `mips_for_texture`.
            levels: 9,
            mip_filter: 1,
            base_format: 0,
            swizzle: 0,
            filter_linear: false,
            addr_mode_u: 0,
            addr_mode_v: 0,
            lod_min: 0,
            gamma: false,
            compressed: None,
            planar_yuv: None,
        }
    }

    /// A texture that passes through compressed is priced at the bytes actually handed to
    /// the driver, NOT at the 4/3-of-level-0 estimate the decode path uses.
    ///
    /// The estimate is what makes this worth asserting: it is derived from `width * height *
    /// bytes_per_texel`, so applying it to a passthrough would price BC3 blocks as RGBA8 and
    /// report a working set four times what is resident. A budget that mis-prices the path
    /// it is meant to be measuring is the same defect the byte accounting was rewritten to
    /// remove, just pointed the other way.
    #[test]
    fn a_compressed_passthrough_is_priced_at_the_bytes_handed_over() {
        let mut t = tex(256, 256, 1, TexelSeam::Rgba8);
        // 256x256 BC3: 64x64 blocks of 16 bytes, plus the chain.
        let blocks = vec![0u8; 64 * 64 * 16 * 4 / 3];
        let n = blocks.len();
        t.compressed = Some(CompressedUpload {
            format: BlockFormat::Bc3,
            width: 256,
            height: 256,
            data: crate::gpu::CompressedData::Cpu(std::sync::Arc::new(blocks)),
            levels: 9,
            transcoded: false,
        });
        assert_eq!(texture_upload_bytes(&t, true), n);
        // And with the device gate shut it is the decode that is priced, not the blocks -
        // the two paths cost wildly different amounts and the flag is the only thing that
        // says which one this texture is on.
        assert_eq!(texture_upload_bytes(&t, false), 256 * 256 * 4 * 4 / 3);
    }

    /// The byte seam is uploaded WITH a mip chain, so it costs about a third more than its
    /// level 0. The flat `w * h * 4` this replaced returned exactly level 0 - which is what
    /// makes this test fail against the old accounting rather than merely differ from it.
    #[test]
    fn the_byte_seam_is_priced_with_its_mip_chain() {
        let level0 = 256 * 256 * 4;
        let got = uploaded_texture_bytes(&tex(256, 256, 1, TexelSeam::Rgba8));
        assert!(
            got > level0,
            "priced {got} for a mipped texture whose level 0 alone is {level0}"
        );
        assert_eq!(got, level0 * 4 / 3);
    }

    /// A texture the guest gave ONE level is sampled from level 0 alone on the device, so no
    /// chain is generated for it - even with `mip_filter` set - and it is priced at level 0.
    #[test]
    fn a_one_level_texture_gets_no_chain_and_is_priced_at_level_zero() {
        let mut t = tex(256, 256, 1, TexelSeam::Rgba8);
        t.levels = 1;
        assert!(!super::mips_for_texture(&t));
        assert_eq!(uploaded_texture_bytes(&t), 256 * 256 * 4);
    }

    /// The half seam is 8 bytes per texel and is uploaded level-0-only. Pricing it as RGBA8
    /// under-charged it by exactly half, on the one seam whose texels are DATA.
    #[test]
    fn the_half_seam_is_priced_at_eight_bytes_per_texel_and_has_no_mips() {
        let as_rgba8 = 512 * 128 * 4;
        let got = uploaded_texture_bytes(&tex(512, 128, 1, TexelSeam::Rgba16Float));
        assert_eq!(got, 512 * 128 * 8, "the half seam is 8 bytes per texel");
        assert_eq!(got, as_rgba8 * 2, "the old accounting was exactly half this");
    }

    /// A cube map uploads six layers and only one was ever priced. Asserted against the
    /// exact expected total rather than `one * 6`: the mip factor is integer arithmetic, so
    /// six faces priced together and one face priced six times differ by a couple of bytes.
    #[test]
    fn every_face_of_a_cube_map_is_priced() {
        let one = uploaded_texture_bytes(&tex(64, 64, 1, TexelSeam::Rgba8));
        let six = uploaded_texture_bytes(&tex(64, 64, 6, TexelSeam::Rgba8));
        assert_eq!(six, 64 * 64 * 6 * 4 * 4 / 3);
        assert!(
            six >= one * 6 - 8 && six <= one * 6 + 8,
            "six faces priced {six}, one face priced {one}"
        );
    }
}

#[cfg(test)]
mod depth_bind_cache_bound_tests {
    use super::{drain_if_at_cap, DEPTH_BG_CACHE_CAP};
    use crate::fasthash::FxHashMap as HashMap;

    /// The defect, reproduced as the access PATTERN that caused it: a race moves the depth
    /// range every frame, so every insertion carries a key never seen before and never
    /// asked for again. Without a bound the map grows once per draw for the whole run -
    /// which is how the GPU process reached 5.8 GB while every picture looked correct.
    ///
    /// Asserting on the CAP rather than on a fixed number is deliberate: the bug was the
    /// absence of any ceiling, so the test has to fail for an unbounded map at whatever
    /// the ceiling happens to be.
    #[test]
    fn a_never_repeating_key_cannot_grow_the_cache_without_bound() {
        let mut map: HashMap<u64, u64> = HashMap::default();
        let mut evicted = Vec::new();
        // Ten times the cap of distinct keys, as a long race would produce.
        for k in 0..(DEPTH_BG_CACHE_CAP as u64 * 10) {
            if !map.contains_key(&k) {
                drain_if_at_cap(&mut map, DEPTH_BG_CACHE_CAP, &mut evicted);
            }
            map.insert(k, k);
            assert!(
                map.len() <= DEPTH_BG_CACHE_CAP,
                "cache reached {} entries against a cap of {DEPTH_BG_CACHE_CAP}",
                map.len()
            );
        }
        // EVERY evicted entry is handed back, not dropped. Each one owns a GPU buffer in
        // the real cache, and one silently dropped is one that can never be destroyed -
        // the whole reason this returns the values instead of clearing.
        assert_eq!(
            evicted.len(),
            DEPTH_BG_CACHE_CAP * 9,
            "evicted entries must all be handed back for destruction"
        );
    }

    /// A cache under its cap must not be disturbed - the bound is a ceiling, not a policy
    /// that throws away a working set. A menu holds one depth range all screen.
    #[test]
    fn a_cache_under_its_cap_is_left_alone() {
        let mut map: HashMap<u64, u64> = HashMap::default();
        for k in 0..(DEPTH_BG_CACHE_CAP as u64 - 1) {
            map.insert(k, k);
        }
        let mut evicted = Vec::new();
        assert!(
            !drain_if_at_cap(&mut map, DEPTH_BG_CACHE_CAP, &mut evicted),
            "cleared while under the cap"
        );
        assert_eq!(map.len(), DEPTH_BG_CACHE_CAP - 1);
        assert!(evicted.is_empty(), "evicted something while under the cap");
        // And it reports the clear when it does happen, so a capture can show it.
        map.insert(u64::MAX, 0);
        assert!(
            drain_if_at_cap(&mut map, DEPTH_BG_CACHE_CAP, &mut evicted),
            "did not clear at the cap"
        );
        assert!(map.is_empty());
        assert_eq!(evicted.len(), DEPTH_BG_CACHE_CAP);
    }
}

#[cfg(test)]
mod content_hash_tests {
    /// The exact shape that shipped: a 150-vertex, 24-byte-stride UI stream whose only
    /// difference is bit 7 of the unorm8 ALPHA at byte 23 of each vertex.
    ///
    /// Byte `24v + 23` is `7 mod 8`, so every one of those bits is bit 63 of a word, and
    /// the word-wise FNV this replaced cancelled them in pairs. 150 is even, so it hashed
    /// the two streams identically and the packed-vertex cache served the wrong mesh.
    #[test]
    fn a_paired_top_bit_flip_does_not_cancel() {
        let mut a = vec![0x5au8; 24 * 150];
        let mut b = a.clone();
        for v in 0..150 {
            a[v * 24 + 23] = 127;
            b[v * 24 + 23] = 255;
        }
        assert_ne!(
            super::fnv64(0xcbf2_9ce4_8422_2325, &a),
            super::fnv64(0xcbf2_9ce4_8422_2325, &b),
            "two vertex streams differing in the top bit of every 8-aligned word hash the \
             same, which is the collision that made a faded text quad render at another \
             frame's alpha"
        );
    }

    /// The general form, so the fix is not read as being about one stride: an EVEN number of
    /// top-bit flips at any 8-byte-aligned positions must change the hash.
    #[test]
    fn top_bit_flips_at_word_boundaries_change_the_hash() {
        let base = vec![0x11u8; 8 * 32];
        for pairs in 1..=8usize {
            let mut v = base.clone();
            for i in 0..(pairs * 2) {
                v[i * 8 + 7] ^= 0x80;
            }
            assert_ne!(
                super::fnv64(0, &base),
                super::fnv64(0, &v),
                "{} paired top-bit flips cancelled",
                pairs
            );
        }
    }

    /// A hash that ignored length, or folded it in only at the end without diffusing it,
    /// would let a truncated stream collide with its own prefix.
    #[test]
    fn length_reaches_the_whole_hash() {
        let v = [0u8; 64];
        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for n in 0..=64 {
            assert!(seen.insert(super::fnv64(0, &v[..n])), "zero-fill of length {n} collided");
        }
    }
}

/// Which passes draw through the half-precision tile shadow - see `GxmRenderer::tile_shadow`.
/// An 8-bit attachment rounding after every draw is what turned an accumulated soft glow into
/// hard concentric bands; a pass the shadow cannot model must keep its attachment.
#[cfg(test)]
mod tile_shadow_tests {
    use super::renderer_pass::tile_shadow_applies;
    use wgpu::{TextureFormat as F, TextureUsages as U};

    const SAMPLED_TARGET: U = U::RENDER_ATTACHMENT.union(U::TEXTURE_BINDING).union(U::COPY_SRC);

    fn rgba8(w: u32, h: u32) -> Option<(F, u32, U, u32, u32)> {
        Some((F::Rgba8Unorm, 1, SAMPLED_TARGET, w, h))
    }

    #[test]
    fn an_eight_bit_pass_with_a_half_precision_self_blender_takes_the_shadow() {
        assert!(tile_shadow_applies(F::Rgba8Unorm, 1, false, rgba8(384, 512), (384, 512), false, true));
        // A gamma pass renders through the sRGB VIEW of the same texture: still an 8-bit pack.
        assert!(tile_shadow_applies(F::Rgba8UnormSrgb, 1, false, rgba8(384, 512), (384, 512), false, true));
        assert!(tile_shadow_applies(
            F::Bgra8Unorm,
            1,
            false,
            Some((F::Bgra8Unorm, 1, SAMPLED_TARGET, 960, 544)),
            (960, 544),
            false,
            true
        ));
    }

    #[test]
    fn no_half_precision_self_blender_keeps_the_attachment() {
        assert!(!tile_shadow_applies(F::Rgba8Unorm, 1, false, rgba8(384, 512), (384, 512), false, false));
    }

    #[test]
    fn a_pass_the_shadow_cannot_model_keeps_the_attachment() {
        // Already wide enough: a float or raw 64-bit target holds the halves itself.
        assert!(!tile_shadow_applies(F::Rgba16Float, 1, false, Some((F::Rgba16Float, 1, SAMPLED_TARGET, 8, 8)), (8, 8), false, true));
        assert!(!tile_shadow_applies(F::Rg32Uint, 1, false, Some((F::Rg32Uint, 1, SAMPLED_TARGET, 8, 8)), (8, 8), false, true));
        // Multisampled, or resolving into another texture.
        assert!(!tile_shadow_applies(F::Rgba8Unorm, 4, false, Some((F::Rgba8Unorm, 4, SAMPLED_TARGET, 8, 8)), (8, 8), false, true));
        assert!(!tile_shadow_applies(F::Rgba8Unorm, 1, true, rgba8(8, 8), (8, 8), false, true));
        // A single-channel alpha surface.
        assert!(!tile_shadow_applies(F::Rgba8Unorm, 1, false, rgba8(8, 8), (8, 8), true, true));
        // No texture to seed from, one that cannot be sampled, or one of another extent.
        assert!(!tile_shadow_applies(F::Rgba8Unorm, 1, false, None, (8, 8), false, true));
        assert!(!tile_shadow_applies(F::Rgba8Unorm, 1, false, Some((F::Rgba8Unorm, 1, U::RENDER_ATTACHMENT, 8, 8)), (8, 8), false, true));
        assert!(!tile_shadow_applies(F::Rgba8Unorm, 1, false, rgba8(16, 16), (8, 8), false, true));
    }
}

/// Which destination readers may share the copy before them, and what each copy must cover -
/// see `DestDirty`. A wrong answer here is a reader blending against a stale pixel, so the
/// last test checks the rule against a per-pixel model of the attachment and its copy.
#[cfg(test)]
mod dest_dirty_tests {
    use super::renderer_pass::{DestDirty, draw_footprint};

    #[test]
    fn a_footprint_is_the_viewport_rounded_outward_cut_by_the_scissor() {
        // Whole attachment.
        assert_eq!(draw_footprint((0.0, 0.0, 960.0, 544.0), (0, 0, 960, 544), 960, 544), [0, 0, 960, 544]);
        // A fractional viewport rounds OUTWARD - a pixel whose sample point sits on its edge
        // can still be covered.
        assert_eq!(draw_footprint((10.5, 20.25, 100.0, 50.5), (0, 0, 960, 544), 960, 544), [10, 20, 111, 71]);
        // The scissor narrows it.
        assert_eq!(draw_footprint((0.0, 0.0, 960.0, 544.0), (100, 50, 20, 10), 960, 544), [100, 50, 120, 60]);
        // A viewport reaching past the attachment is clamped to it.
        assert_eq!(draw_footprint((-50.0, -10.0, 2000.0, 2000.0), (0, 0, 960, 544), 960, 544), [0, 0, 960, 544]);
        // A zero-area scissor (REGION_CLIP_ALL) writes nothing.
        let f = draw_footprint((0.0, 0.0, 960.0, 544.0), (0, 0, 0, 0), 960, 544);
        assert!(f[0] >= f[2] || f[1] >= f[3]);
        // A NaN viewport cannot be bounded and reaches the whole attachment.
        assert_eq!(draw_footprint((f32::NAN, 0.0, f32::NAN, 544.0), (0, 0, 960, 544), 960, 544), [0, 0, 960, 544]);
    }

    #[test]
    fn the_first_reader_of_a_pass_always_takes_a_whole_copy() {
        let mut d = DestDirty::new();
        assert!(d.needs_copy([0, 0, 1, 1]));
        assert_eq!(d.take_copy(), None, "the copy texture holds another pass's pixels on entry");
    }

    #[test]
    fn a_reader_clear_of_everything_written_since_the_copy_shares_it() {
        let mut d = DestDirty::new();
        d.take_copy();
        d.wrote([0, 0, 100, 100]);
        // Disjoint: shares. Touching edges are disjoint - the far edges are exclusive.
        assert!(!d.needs_copy([100, 0, 200, 100]));
        assert!(!d.needs_copy([0, 100, 100, 200]));
        // Overlapping by one pixel: cuts.
        assert!(d.needs_copy([99, 99, 200, 200]));
        // A refresh copy covers only what was written since the last one.
        d.wrote([300, 300, 310, 320]);
        assert_eq!(d.take_copy(), Some([0, 0, 310, 320]));
        // ...and with nothing written in between, an empty one.
        assert_eq!(d.take_copy(), Some([0, 0, 0, 0]));
    }

    #[test]
    fn merging_past_the_cap_only_grows_the_dirty_set() {
        let mut d = DestDirty::new();
        d.take_copy();
        // Many small disjoint rects along a diagonal, then a probe in a gap between two of them:
        // kept apart it shares; once merged into the bounding box it must cut, never the reverse.
        for i in 0..200u32 {
            d.wrote([i * 4, i * 4, i * 4 + 2, i * 4 + 2]);
        }
        assert!(d.needs_copy([2, 0, 4, 2]), "after the merge the gap is inside the bounding box");
    }

    /// Per-pixel model: the attachment and the copy each carry a VERSION per pixel. Every draw
    /// bumps the versions it covers in the attachment; every copy brings the copy up to date over
    /// the region `take_copy` names; and every reader must find the copy current over its whole
    /// footprint - which is exactly "reads what a fresh copy would have handed it".
    #[test]
    fn no_reader_ever_reads_a_stale_pixel() {
        const W: u32 = 24;
        const H: u32 = 16;
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = |n: u32| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as u32
        };
        // Readers that shared a copy, over every pass: the model must exercise the sharing it
        // checks, or it proves nothing.
        let mut shared = 0;
        for _pass in 0..400 {
            let mut att = vec![0u32; (W * H) as usize];
            // The copy starts as some OTHER pass left it: never current.
            let mut copy = vec![u32::MAX; (W * H) as usize];
            let mut d = DestDirty::new();
            let (mut cuts, mut readers) = (0, 0);
            for draw in 1..=40u32 {
                let x = rnd(W + 4) as f32 - 2.0 + rnd(4) as f32 * 0.25;
                let y = rnd(H + 4) as f32 - 2.0 + rnd(4) as f32 * 0.25;
                let vp = (x, y, rnd(W) as f32 + 0.5, rnd(H) as f32 + 0.5);
                let sc = if rnd(3) == 0 { (rnd(W), rnd(H), rnd(W), rnd(H)) } else { (0, 0, W, H) };
                let f = draw_footprint(vp, sc, W, H);
                let reader = rnd(2) == 0;
                if reader {
                    readers += 1;
                    if d.needs_copy(f) {
                        cuts += 1;
                        let [x0, y0, x1, y1] = d.take_copy().unwrap_or([0, 0, W, H]);
                        for py in y0..y1.min(H) {
                            for px in x0..x1.min(W) {
                                copy[(py * W + px) as usize] = att[(py * W + px) as usize];
                            }
                        }
                    }
                    for py in f[1]..f[3] {
                        for px in f[0]..f[2] {
                            let i = (py * W + px) as usize;
                            assert_eq!(copy[i], att[i], "reader {draw} read a stale pixel ({px},{py})");
                        }
                    }
                }
                for py in f[1]..f[3] {
                    for px in f[0]..f[2] {
                        att[(py * W + px) as usize] = draw;
                    }
                }
                d.wrote(f);
            }
            assert!(cuts <= readers);
            shared += readers - cuts;
        }
        assert!(shared > 1000, "only {shared} readers shared a copy - the model is not exercising the rule");
    }

    #[test]
    fn refining_asks_only_where_the_answer_can_change() {
        let exact = |d: usize| -> [u32; 4] { [d as u32 * 10, 0, d as u32 * 10 + 5, 5] };
        let mut asked = Vec::new();
        let mut d = DestDirty::new();
        // The pass's first reader cuts without asking anything.
        assert!(d.needs_copy_refining(0, [0, 0, 100, 100], &mut |i| {
            asked.push(i);
            exact(i)
        }));
        d.take_copy();
        d.wrote_coarse([0, 0, 100, 100], 0);
        d.wrote_coarse([0, 0, 100, 100], 1);
        d.wrote_coarse([0, 0, 100, 100], 2);
        // Coarse footprint clear of everything: shares, nothing asked.
        assert!(!d.needs_copy_refining(9, [200, 200, 300, 300], &mut |i| {
            asked.push(i);
            exact(i)
        }));
        assert!(asked.is_empty());
        // Reader 3 sits at 30..35: it is refined, then the writers newest first - and none meets it.
        assert!(!d.needs_copy_refining(3, [0, 0, 100, 100], &mut |i| {
            asked.push(i);
            exact(i)
        }));
        assert_eq!(asked, vec![3, 2, 1, 0]);
        asked.clear();
        // A reader over writer 1 cuts on it. The newer coarse writer 4 is refined on the way and
        // clears it; the writers refined before are not asked twice.
        d.wrote_coarse([0, 0, 100, 100], 4);
        assert!(d.needs_copy_refining(7, [0, 0, 100, 100], &mut |i| {
            asked.push(i);
            if i == 7 { [12, 0, 13, 1] } else { exact(i) }
        }));
        assert_eq!(asked, vec![7, 4]);
        // The refresh copy covers the refined writers.
        assert_eq!(d.take_copy(), Some([0, 0, 45, 5]));
        // A coarse writer nobody asked about is copied whole.
        d.wrote_coarse([0, 0, 100, 100], 5);
        assert_eq!(d.take_copy(), Some([0, 0, 100, 100]));
    }

    /// The per-pixel model again, with every draw's TRUE coverage a random part of its coarse
    /// footprint (possibly nothing), revealed only through `needs_copy_refining`.
    #[test]
    fn refined_footprints_never_read_a_stale_pixel() {
        const W: u32 = 24;
        const H: u32 = 16;
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move |n: u32| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as u32
        };
        let mut shared = 0;
        for _pass in 0..400 {
            let mut att = vec![0u32; (W * H) as usize];
            let mut copy = vec![u32::MAX; (W * H) as usize];
            let mut d = DestDirty::new();
            let mut fine = Vec::new();
            for draw in 0..40usize {
                let vp = (rnd(W) as f32 - 2.0, rnd(H) as f32 - 2.0, rnd(W) as f32 + 0.5, rnd(H) as f32 + 0.5);
                let c = draw_footprint(vp, (0, 0, W, H), W, H);
                let f = if c[0] >= c[2] || c[1] >= c[3] || rnd(4) == 0 {
                    [0; 4]
                } else {
                    let x0 = c[0] + rnd(c[2] - c[0]);
                    let y0 = c[1] + rnd(c[3] - c[1]);
                    [x0, y0, x0 + 1 + rnd(c[2] - x0), y0 + 1 + rnd(c[3] - y0)]
                };
                fine.push(f);
                if rnd(2) == 0 {
                    let fine_now = fine.clone();
                    if d.needs_copy_refining(draw, c, &mut |i| fine_now[i]) {
                        let [x0, y0, x1, y1] = d.take_copy().unwrap_or([0, 0, W, H]);
                        for py in y0..y1.min(H) {
                            for px in x0..x1.min(W) {
                                copy[(py * W + px) as usize] = att[(py * W + px) as usize];
                            }
                        }
                    } else {
                        shared += 1;
                    }
                    for py in f[1]..f[3] {
                        for px in f[0]..f[2] {
                            let i = (py * W + px) as usize;
                            assert_eq!(copy[i], att[i], "reader {draw} read a stale pixel ({px},{py})");
                        }
                    }
                }
                for py in f[1]..f[3] {
                    for px in f[0]..f[2] {
                        att[(py * W + px) as usize] = draw as u32 + 1;
                    }
                }
                if rnd(2) == 0 { d.wrote(f) } else { d.wrote_coarse(c, draw) }
            }
        }
        assert!(shared > 1000, "only {shared} readers shared a copy - the model is not exercising the rule");
    }
}

/// The per-pass price on vertex footprints - see `DestGeomGate`.
#[cfg(test)]
mod dest_geom_gate_tests {
    use super::renderer_pass::DestGeomGate;

    #[test]
    fn a_pass_that_does_not_earn_its_footprints_skips_them_then_probes_again() {
        let mut g = DestGeomGate::new();
        let probe = g.begin().expect("a first sight probes");
        // Four cuts saved for 45 ms: not worth it.
        g.end(4, 45_000.0);
        let skipped = std::iter::from_fn(|| Some(g.begin())).take_while(Option::is_none).count();
        assert!(skipped > 0, "an unprofitable pass must skip");
        // The skip ran out and the next encode probed (that `begin` was consumed above) - with
        // the probe budget again, and a second failure skips for longer.
        g.end(0, probe);
        let longer = std::iter::from_fn(|| Some(g.begin())).take_while(Option::is_none).count();
        assert!(longer > skipped, "{longer} <= {skipped}");
    }

    #[test]
    fn a_pass_that_earns_them_may_spend_what_it_earned() {
        let mut g = DestGeomGate::new();
        g.begin().unwrap();
        // 124 cuts saved for 1.7 ms.
        g.end(124, 1_700.0);
        let b = g.begin().expect("a profitable pass keeps its footprints");
        assert!(b >= 124.0 * 50.0, "budget {b}");
        // Nothing to refine is not a loss - and not evidence: the earned budget stands.
        g.end(0, 0.0);
        assert_eq!(g.begin(), Some(b));
    }
}
