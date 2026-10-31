//! [`build_gxp_pipeline`]: link a guest shader pair and build its pipeline - the vertex
//! plan, the bind-group layouts, and the guest's own fixed-function state baked in.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn build_gxp_pipeline<P>(
    device: &wgpu::Device,
    color_format: wgpu::TextureFormat,
    // A single-channel ALPHA target - see `alpha_single_color_format`.
    alpha_single: bool,
    // Samples per pixel of the attachments this pipeline will be bound against. A
    // pipeline's multisample state must match its pass exactly, so this is part of the
    // cache key - see `GxpPrepared::samples`.
    samples: u32,
    // The guest's `SceGxmCullMode` for this draw. Part of the cache key because it is baked
    // into the pipeline's primitive state, and a title sets it PER DRAW.
    cull: u32,
    gxp: &GxpRecompile,
    key: u64,
    zfix: ZFix,
    yflip: bool,
    solid: bool,
    nodepth: bool,
    noblend: bool,
    // This draw's colour is provably the DESTINATION UNCHANGED (`draw_colour_is_noop`) AND
    // it writes depth, so the pipeline keeps the vertex stage and the depth rule and drops
    // the colour work - see the fragment state below. Part of the cache key through
    // `raster_key`. (A no-op draw that writes no depth is never issued at all, so it never
    // reaches here.)
    depth_only: bool,
    // Fragment texture units (bit = unit) bound to a 64-bit target this frame rendered as
    // RAW words - see `LinkOptions::raw_units`. Part of the pipeline cache key.
    raw_units: u64,
    // Units bound to a 32-bit float texture (bit u fragment, bit 32+u vertex) - see
    // `SamplerDim::TwoF32`. Part of the pipeline cache key through the raster key.
    f32_units: u64,
    // Compiled modules by pair - see `GxpLive::modules`.
    modules: &mut HashMap<u64, wgpu::ShaderModule>,
    // Makes the pipeline from its finished descriptor - `create_render_pipeline` for a
    // build in the frame, `create_render_pipeline_async` for one started ahead of it.
    create: impl FnOnce(&wgpu::Device, &wgpu::RenderPipelineDescriptor) -> P,
) -> Option<GxpPipelineOf<P>> {
    let _owner = crate::heap::owner_scope(crate::heap::Owner::Pipeline);
    // Settled before the first link: one gates a rewrite of the program, the other whether
    // the module declares a destination texture at all.
    let _ = crate::gpu::gxp_dest_blend_lowering();
    let _ = crate::gpu::gxp_dest_enabled();
    let debug = std::env::var_os("VITASLOP_GXP_DEBUG").is_some();
    // Arm the evidence BEFORE anything that can refuse this pair: a refusal that names a
    // pair nobody can reconstruct costs a play session per attempt. See `blob_evidence`.
    arm_fallback_blobs(key, gxp.vprog.clone(), gxp.fprog.clone());
    report_branches(key, gxp);
    // >>> OUR OWN HALF OF THE TRANSLATION, TIMED SEPARATELY FROM THE DRIVER'S.
    //
    // `create_shader_module` and `create_render_pipeline` are timed below, and between them
    // they are the whole in-frame shader cost a run reports - which invites reading that
    // total as "the driver is slow" when a third of the work in this function is OURS:
    // decoding both USSE programs, building the IR and emitting ~27 KB of WGSL text. Those
    // have completely different fixes (a Rust profile against emitting fewer operations),
    // so a number that adds them together names neither.
    let t_link = Stopwatch::start();
    let dual_source = draw_dual_source(gxp);
    // A pass into a 64-bit surface takes the fragment's raw register pair - see
    // `raw64_color_format`. The link refuses the shapes an integer attachment cannot serve.
    let raw64_output = color_format == RAW64_FORMAT;
    // A program that blends for itself in a MULTISAMPLED pass reads its own SAMPLE of the
    // destination - see `LinkOptions::dest_multisampled`.
    let dest_ms = samples > 1 && fragment_reads_dest(&gxp.fprog);
    // >>> WHAT THE GUEST BOUND, HANDED TO THE LINK so it can decide which attributes are
    // >>> INTEGER-fetched - see `VertexAttribute::int_fetch` and `integer_vertex_format`.
    //
    // Sorted by base lane so the list is a function of the layout and not of the order the
    // guest happened to declare it in: it is folded into the module cache key below, and a
    // key that changed with declaration order would compile the same module twice.
    let guest_attrs = gxp_guest_attr_spec(gxp);
    // >>> THIS DRAW'S COLOUR MASK, READ BEFORE THE LINK BECAUSE THE LINK NEEDS IT - see
    // >>> `LinkOptions::colour_output_masked_off`.
    //
    // The same two guest facts the pipeline's `write_mask` is built from further down, and
    // deliberately the same expression: a draw whose fragment program the guest DISABLED
    // (`sceGxmSetFrontFragmentProgramEnable`) or whose blend info masks every channel
    // cannot put its colour on screen, so a fragment program that computes no colour is
    // reproduced exactly by one that computes anything at all.
    //
    // NOT claimed under `solid` or `noblend`: both are diagnostic knobs that force the
    // mask back to ALL below, and a link that assumed the colour was discarded while the
    // pipeline went on to write it would be a silently wrong picture under a knob whose
    // whole purpose is to make pictures readable.
    let colour_output_masked_off = !solid
        && !noblend
        && match gxp.fragment_program_enabled {
            true => gxm_color_mask(gxp.blend_state[0]).is_empty(),
            false => true,
        };
    // MEMOISED on exactly these inputs - a pair the shader patcher named was translated at
    // the patcher, on the title's own load (see `vitaslop_gxp_shader::link_programs_memo`).
    let link_opts = vitaslop_gxp_shader::LinkOptions {
        dual_source,
        raw_units,
        raw64_output,
        // Only for a program that reads the destination: the option changes nothing for any
        // other, and setting it would miss the patcher-time memo for every MSAA draw.
        dest_multisampled: dest_ms,
        guest_attrs: guest_attrs.clone(),
        colour_output_masked_off,
        alpha_to_red: alpha_single && !raw64_output,
        patched_against: (!gxp.fprog_patched_vprog.is_empty()).then(|| gxp.fprog_patched_vprog.clone()),
    };
    // A translation paid IN the frame names its pair and why the memo missed (never
    // translated, or which option the patcher-time guess got wrong) - "build ms ours 401"
    // on a stall line is not actionable until it says that.
    let miss = vitaslop_gxp_shader::memo_miss_reason(&gxp.vprog, &gxp.fprog, &link_opts);
    let linked = {
        let _owner = crate::heap::owner_scope(crate::heap::Owner::Shader);
        vitaslop_gxp_shader::link_programs_memo(&gxp.vprog, &gxp.fprog, link_opts)
    };
    if let Some(why) = &miss
        && t_link.ms() > 15.0
    {
        report_status!("slow in-frame translation: pair {key:016x} {:.0} ms - memo miss: {why}", t_link.ms());
    }
    // `VITASLOP_DUMP_SLOW_PAIRS=<dir>` (native): every pair translated IN the frame (a memo
    // miss), as its two containers, so a slow link can be profiled offline
    // (`tests/corpus.rs` `profile_slow_pairs`) by the key the slow line above names, on
    // whichever engine named it.
    #[cfg(not(target_arch = "wasm32"))]
    if miss.is_some()
        && let Some(dir) = std::env::var_os("VITASLOP_DUMP_SLOW_PAIRS")
    {
        let dir = std::path::Path::new(&dir);
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(dir.join(format!("{key:016x}_vert.gxp")), &gxp.vprog[..]);
        let _ = std::fs::write(dir.join(format!("{key:016x}_frag.gxp")), &gxp.fprog[..]);
    }
    let linked = match linked {
        Ok(l) => l,
        Err(e) => {
            crate::gpu::add_build_ms(&crate::gpu::PIPE_LINK_US, t_link.ms());
            // >>> AND THE DRAW STATE THE LINK WAS GIVEN, because a refusal that a per-draw
            // >>> option was supposed to LIFT has to say whether the option was set.
            //
            // `report_zero_color_mask` cannot answer this: it is computed far below, after
            // the link, and a refused pair returns HERE - so a run of a title whose pairs
            // are all refused reports ZERO zero-masks and that number means nothing at all.
            // A reader (this one) took it for evidence that the masks were non-zero.
            report_fallback(
                key,
                &format!(
                    "link failed: {e} [this draw: colour {}, fragment program {}]",
                    if colour_output_masked_off { "MASKED OFF" } else { "written" },
                    if gxp.fragment_program_enabled { "enabled" } else { "DISABLED" },
                ),
            );
            return None;
        }
    };

    // `VITASLOP_GXP_WGSL_LOG=1`: every module linked in this run, whole, at WARN - the one
    // way to read the shader a DEVICE compiles (a runner live job forwards the console).
    if crate::knobs::flag("VITASLOP_GXP_WGSL_LOG") {
        tracing::warn!(target: "vitaslop::gxp", "gxp wgsl key {key:x}
{}", linked.wgsl);
    }
    // Vertex layout: each linked attribute (@location L, base lane B) is fed by the guest
    // stream attribute whose reg_index == B. We repack it to tightly-packed `Float32xN`
    // (converting F16/U8N/etc. on the CPU) so wgpu's vertex-format gaps (no Float16x3, no
    // Unorm8x3, ...) never block a draw, and the shader (which reads f32) gets exact values.
    let mut wattrs: Vec<wgpu::VertexAttribute> = Vec::with_capacity(linked.vertex_bindings.attributes.len());
    let mut repack: Vec<RepackAttr> = Vec::with_capacity(linked.vertex_bindings.attributes.len());
    let mut packed_offset: u32 = 0;
    // The SAME layout expressed against the guest's own row, when every attribute of it can
    // be (see `passthrough_vertex_format`). Built beside the repack rather than instead of
    // it: whichever plan is chosen, the other stays available as the A/B arm, and a single
    // attribute that cannot be bound natively takes the whole stream back to the repack -
    // one buffer has one stride, so this is all-or-nothing per pipeline.
    let mut native: Vec<wgpu::VertexAttribute> = Vec::with_capacity(wattrs.capacity());
    // >>> TWO DIFFERENT QUESTIONS, TWO DIFFERENT REASONS, and they must not be pooled.
    //
    // `no_native` is why an ATTRIBUTE cannot keep the guest's component format and has to be
    // widened to f32 - the ~7x, and the thing worth fixing. `no_row` is why the guest's ROW
    // cannot be BOUND as it stands, which costs a byte copy and nothing else. An offset that
    // WebGPU will not accept is only ever the second; reporting it as the first sends the
    // reader after the wrong thing, which a diagnostic must not do.
    //
    // The FIRST reason of each, not a list: the remedy for each is separate work, and a
    // list of every attribute reads as noise.
    let mut no_native: Option<String> = None;
    let mut no_row: Option<String> = None;
    // What `packed_stride` would be if every attribute kept the guest's own component
    // format. Only the DIFFERENCE is used, and only to price the expansion in uploaded
    // bytes - see `arena_bytes_expand_excess`.
    let mut native_packed_offset: u32 = 0;
    // The fill knob overrides the linker's per-lane answer on the CPU, inside `attr_fill`.
    // Nothing the hardware's vertex fetch does can honour that, so under it the f32
    // expansion is the only plan that means what the knob says - and an A/B whose two arms
    // silently ignore one another's knobs is not an A/B.
    let fill_knob = crate::knobs::var_os("VITASLOP_GXP_ATTR_FILL").is_some();
    // The arm back restores the whole of the old behaviour, which is BOTH halves of this:
    // no guest row bound, and every component expanded to f32 by the CPU.
    let native_ok = crate::gpu::gxp_vertex_passthrough() && !fill_knob;
    for a in &linked.vertex_bindings.attributes {
        let ga: &GxpAttr = match gxp.attributes.iter().find(|g| g.reg_index as u32 == a.base_lane) {
            Some(g) => g,
            None => {
                report_fallback(
                    key,
                    &format!(
                        "no guest attribute for linked @location {} base_lane {} (guest reg_indices {:?})",
                        a.location,
                        a.base_lane,
                        gxp.attributes.iter().map(|g| g.reg_index).collect::<Vec<_>>()
                    ),
                );
                return None;
            }
        };
        let comps = ga.components.clamp(1, 4);
        // The SHADER's declared width, which is what its PA loads read. When the guest binds
        // FEWER components than that, the surplus registers are fed a fill - and which fill
        // is a question about the guest's vertex fetch that nothing here had ever asked.
        // Widening the packed slot to the declared width and writing the fill ourselves makes
        // it an A/B instead of whatever the graphics API happens to do. The DEFAULT is 1.0,
        // not the API's `(0, 0, 0, 1)` - see `attr_fill`, which records the title that lost
        // its colour to the zero. `VITASLOP_GXP_ATTR_FILL=api` restores the old reading.
        // >>> AND WHETHER THE MODULE READS ANY LANE ABOVE WHAT THE GUEST BOUND AT ALL.
        //
        // When the link was told the guest's component count it BAKES the surplus lanes as
        // constants (`VertexAttribute::guest_components`), so a row - ours or the guest's -
        // no longer has to carry them.
        let baked = a.guest_components.is_some_and(|b| b >= comps as u32);
        // The SHADER's declared width, narrowed to what the guest binds when the surplus is
        // baked: a lane the module does not read is a lane no row owes.
        let slots = if baked { comps } else { (a.components as u8).clamp(comps, 4) };
        // The LINKER's per-lane answer (see `attr_fill`), resolved here into plain floats.
        let surplus_fill: [f32; 4] = std::array::from_fn(|c| a.surplus_fill[c].value());
        if slots > comps {
            // >>> THE GUEST'S ATTRIBUTE FORMAT, because it is the only variable left that
            // >>> could make ONE rule right for two titles that need opposite fills.
            //
            // MEASURED both ways on this machine: a racer's cows are correct at a fill
            // of 1.0 and turn BRIGHT GREEN/CYAN under the API's `(0,0,0,1)`; a fighting title's effect
            // meshes do the reverse - `Input.Color` with blue pinned to 1.0 is the
            // saturated mass a user photographed. `component_count` is the guest's own word
            // (read straight off `SceGxmVertexAttribute`), so this is not a case of
            // mis-reading how much is bound: both titles really do declare wider than they
            // bind, and a constant cannot serve both. If the two split on FORMAT -
            // normalised integer against float - the rule keys on that and is justified by
            // both titles at once instead of by either alone.
            report_attr_fill(
                key,
                a.location,
                a.base_lane,
                comps,
                slots,
                ga.gxm_format,
                surplus_fill,
            );
        }
        // >>> CAN THE HARDWARE DO THIS ATTRIBUTE'S CONVERSION INSTEAD OF THE CPU?
        //
        // Two questions, and both have to be YES:
        //   * the FORMAT converts to `f32` the way `read_attr_component` does (and the
        //     narrowest WebGPU format for it may fetch ONE component more than the guest
        //     binds - see `widest_passthrough_format`);
        //   * every lane the guest does NOT supply is fed what the hardware would feed it.
        //     The f32 expansion writes lanes `comps..slots` itself, from the LINKER's
        //     per-lane answer; a native fetch cannot, so the two agree only where that
        //     answer IS the hardware's - or where nothing reads the lane at all
        //     (`attrflow::Fill::Unobserved`), which is the common case and the one that
        //     decides most layouts. MEASURED on a golf title: 102 pipelines were held back
        //     by a lane that turned out to be unread, because the analysis said "unread"
        //     and "must be 1.0" with the same word. See `attr_fill` for the title that lost
        //     its colour to a wrong constant here - this is the one place the two plans
        //     could differ in VALUE rather than in cost, so it is checked per lane.
        //
        // Note what is NOT asked here: the guest's byte offset and stride. This decides the
        // format of the attribute's slot in a row THIS code lays out, so it can always give
        // it a legal offset. Binding the guest's own row is a further, stricter question,
        // asked once below.
        // >>> AN INTEGER-FETCHED ATTRIBUTE IS ALREADY DECIDED, AND IT IS NOT OPTIONAL.
        //
        // The MODULE declares `vec4<u32>`/`vec4<i32>` for it, so this layout must bind an
        // integer format or WebGPU refuses the pipeline. The two checks below - a lane the
        // widened fetch peeks at, a surplus lane wanting our constant - are exactly the
        // proof the LINK already made before setting the flag (every lane above the guest's
        // binding is unobserved), so re-asking them here could only disagree with it.
        let int_fmt = a.int_fetch.and_then(|_| integer_vertex_format(ga.gxm_format, comps));
        // Neither check below has a subject once the surplus is BAKED: a lane the widened
        // fetch peeks at is not read, and a lane that wants our constant already has it in
        // the shader. Those two are exactly what held 46 of this title's pipelines - about
        // four fifths of its draws - on the repack.
        let native_fmt = match int_fmt {
            Some(f) => Some(f),
            None if baked => native_ok
                .then(|| widest_passthrough_format(ga.gxm_format, comps))
                .flatten(),
            None => match native_ok.then(|| widest_passthrough_format(ga.gxm_format, comps)).flatten() {
            None if !native_ok => None,
            None => {
                no_native.get_or_insert_with(|| {
                    format!(
                        "@location {} is GXM format {} x{} components, which has no WebGPU \
                         vertex format",
                        a.location, ga.gxm_format, comps
                    )
                });
                None
            }
            Some((f, fetched)) => {
                // Lanes the FORMAT fetches but the guest never bound. In the packed row they
                // are zero-filled; binding the guest's row gives them whatever bytes follow.
                // Either way they are not this attribute's data, so nothing may read them.
                let peeked = (comps..fetched).find(|c| a.surplus_fill[*c as usize].observed());
                // Lanes above what the format fetches: the hardware's own `(0, 0, 0, 1)`.
                let bad_fill = (fetched.max(comps)..slots).find(|c| {
                    a.surplus_fill[*c as usize].observed()
                        && surplus_fill[*c as usize] != if *c == 3 { 1.0 } else { 0.0 }
                });
                if let Some(c) = bad_fill {
                    no_native.get_or_insert_with(|| {
                        format!(
                            "@location {} lane {c} is filled with {} and the hardware would \
                             supply {}",
                            a.location,
                            surplus_fill[c as usize],
                            if c == 3 { 1.0 } else { 0.0 }
                        )
                    });
                    None
                } else if let Some(c) = peeked {
                    no_native.get_or_insert_with(|| {
                        format!(
                            "@location {} is bound {comps} wide and the narrowest format for \
                             GXM {} fetches {fetched}, so lane {c} is not its data - and \
                             something READS it",
                            a.location, ga.gxm_format
                        )
                    });
                    None
                } else {
                    Some((f, fetched))
                }
            }
            },
        };
        // >>> THE PACKED ROW KEEPS THE GUEST'S OWN COMPONENT FORMAT WHEREVER IT CAN, and
        // >>> that is where the 7x goes.
        //
        // The expansion this path was named for is not the COPY, it is the WIDTH: writing
        // every component as a 4-byte f32 turns an `F16x2` texcoord into 8 bytes and a
        // `U8N` colour into 16. Keeping the guest's format makes the slot the guest's own
        // size and hands the conversion to the vertex fetch, which does it for free - so
        // the row this builds is about as wide as the guest's, and the repack becomes a
        // byte copy per attribute instead of a float conversion per component.
        //
        // Per ATTRIBUTE, not per row: an attribute whose surplus lane really does need this
        // renderer's constant keeps its f32 slot beside the others, because the row is ours
        // to lay out and nothing requires one rule for all of it.
        let (format, slot_bytes, copy_bytes) = match native_fmt {
            // The copy is what the GUEST BINDS, never what the format fetches: a widened
            // format's extra lane is left as the row's zero fill rather than filled from
            // the bytes that happen to follow, so the packed row does not depend on what
            // its neighbour is - and a guest buffer too short to hold the attribute at all
            // zero-fills exactly as `read_attr_component` does.
            Some((f, _)) => (
                f,
                (f.size() as u32).next_multiple_of(4),
                comps as u32 * attr_component_size(ga.gxm_format) as u32,
            ),
            None => {
                let f = match slots {
                    1 => wgpu::VertexFormat::Float32,
                    2 => wgpu::VertexFormat::Float32x2,
                    3 => wgpu::VertexFormat::Float32x3,
                    _ => wgpu::VertexFormat::Float32x4,
                };
                (f, slots as u32 * 4, 0)
            }
        };
        // The same slot sized in the GUEST's format, whether or not this attribute could
        // actually keep it. A slot is 4-aligned either way, so the two strides differ by
        // exactly the widening.
        native_packed_offset += if native_fmt.is_some() {
            slot_bytes
        } else {
            (comps as u32 * attr_component_size(ga.gxm_format) as u32).max(1).next_multiple_of(4)
        };
        wattrs.push(wgpu::VertexAttribute { format, offset: packed_offset as u64, shader_location: a.location });
        // The guest's own row, bound as it stands, needs everything the packed row needs AND
        // the guest's byte offset to be one WebGPU accepts for the format, with the whole
        // (possibly widened) read inside the row.
        if let Some((f, _)) = native_fmt {
            let goff = ga.offset as u32;
            // WebGPU's own rule, and it is the FORMAT's size that it caps at 4 - not the
            // component's. `Unorm8x4` is therefore 4-aligned even though its components are
            // single bytes, and reading that as "one-byte components, so any offset"
            // produces a pipeline the driver refuses at creation - which on this path is a
            // pair dropped for the rest of the run.
            let align = (f.size() as u32).min(4);
            if !goff.is_multiple_of(align) {
                no_row.get_or_insert_with(|| {
                    format!("@location {} sits at guest offset {goff}, not a multiple of {align}", a.location)
                });
            } else if goff + f.size() as u32 > gxp.vertex_stride {
                no_row.get_or_insert_with(|| {
                    format!(
                        "@location {} reaches byte {} of a {}-byte guest row",
                        a.location,
                        goff + f.size() as u32,
                        gxp.vertex_stride
                    )
                });
            } else {
                native.push(wgpu::VertexAttribute {
                    format: f,
                    offset: goff as u64,
                    shader_location: a.location,
                });
            }
        }
        repack.push(RepackAttr {
            guest_offset: ga.offset as u32,
            gxm_format: ga.gxm_format,
            components: comps,
            slots,
            fill: surplus_fill,
            packed_offset,
            copy_bytes,
        });
        packed_offset += slot_bytes;
    }
    let packed_stride = packed_offset.max(4);
    let packed_stride_native = native_packed_offset.max(4).min(packed_stride);

    // >>> AND THE ROW ITSELF, which is a property of the STREAM rather than of any one
    // >>> attribute, so it is asked once here.
    //
    // `gxp.vertices` is already ONE interleaved row per vertex: `gather_into_scratch` tiles
    // the guest's bound streams into it with no gap and no overlap, rebasing each stream's
    // attribute offsets and repeating a per-instance or stride-0 stream's single row into
    // every vertex. That is exactly the shape a WebGPU vertex buffer wants, which is why
    // this can be a binding rather than a conversion.
    let stride_ok = gxp.vertex_stride >= 4
        && gxp.vertex_stride.is_multiple_of(4)
        && gxp.vertex_stride <= MAX_VERTEX_ARRAY_STRIDE;
    let passthrough = native_ok
        && stride_ok
        && no_native.is_none()
        && native.len() == wattrs.len()
        && !wattrs.is_empty();
    // >>> THREE OUTCOMES, AND THEY COST DIFFERENT THINGS - so the run says which it got.
    //
    //   * PASSTHROUGH: the guest's row is bound as it stands. No copy at all.
    //   * a packed row that is entirely NATIVE: one byte copy per attribute, a row about as
    //     wide as the guest's. This is what a legal-format-but-illegal-offset layout gets,
    //     and it keeps the whole byte win while giving up the copy.
    //   * a packed row with EXPANDED slots: the f32 widening, ~7x on the attributes that
    //     take it, which is the thing being removed. This is the only one that needs a
    //     reason, and it is per ATTRIBUTE rather than per pipeline.
    //
    // All at debug except the last, which is censused: an ordinary run's panel should not
    // fill with layouts that are working.
    let expanded = repack.iter().filter(|r| r.copy_bytes == 0).count();
    // >>> COUNTED, ALWAYS, whatever the panels do with the lines below. See
    // >>> `VertexPlanCensus` for the run in which the reports and the per-draw counter said
    // >>> opposite things and neither could be checked.
    if wattrs.is_empty() {
        note_vertex_plan("none", None);
    } else if expanded > 0 {
        note_vertex_plan("expand", no_native.as_deref());
    } else if passthrough {
        note_vertex_plan("passthrough", None);
    } else if !stride_ok {
        note_vertex_plan(
            "copy",
            Some(&format!(
                "the guest row is {} bytes, which WebGPU cannot use as a stride",
                gxp.vertex_stride
            )),
        );
    } else {
        note_vertex_plan("copy", no_row.as_deref());
    }
    if crate::gpu::gxp_vertex_passthrough() && !wattrs.is_empty() {
        if expanded > 0 {
            let why = if fill_knob {
                "VITASLOP_GXP_ATTR_FILL overrides the fill on the CPU".to_string()
            } else {
                no_native.clone().unwrap_or_else(|| "no attributes".to_string())
            };
            report_vertex_repack(key, &format!("{expanded} of {} attributes expand to f32: {why}", repack.len()));
        } else if passthrough {
            // At STATUS, so an A/B can be read as live from the LOGS - see the function.
            report_vertex_plan_ok(&format!(
                "pair {key:016x}: binds the guest's own {}-byte row, {} attribute(s), no copy at all",
                gxp.vertex_stride,
                wattrs.len()
            ));
        } else {
            // The same control, for the outcome that still copies but no longer widens.
            report_vertex_plan_ok(&format!(
                "pair {key:016x}: copies a {}-byte guest row into a {packed_stride}-byte one",
                gxp.vertex_stride
            ));
            report!(
                "gxp vertex native repack: pair {key:016x} copies a {}-byte guest row into a \
                 {packed_stride}-byte row, {} attribute(s), every one in the guest's own \
                 format - {}",
                gxp.vertex_stride,
                wattrs.len(),
                if stride_ok {
                    no_row.clone().unwrap_or_else(|| "the row itself is fine".to_string())
                } else {
                    format!("the guest row is {} bytes, which WebGPU cannot use as a stride", gxp.vertex_stride)
                }
            );
        }
    }
    // The vertex buffer's own layout follows whichever plan was chosen.
    let (wattrs, array_stride) = if passthrough {
        (native, gxp.vertex_stride)
    } else {
        (wattrs, packed_stride)
    };

    // Decisive diagnostic (`VITASLOP_GXP_INTERP`): run the recompiled vertex shader on the
    // CPU interpreter with the real captured SA + the first vertex's attributes, and print
    // the clip output o[0..3]. If the recompiler math + uniforms are right, this is a
    // sensible clip position; if it is off-screen/NaN, the problem is upstream of the GPU.
    if std::env::var_os("VITASLOP_GXP_INTERP").is_some()
        && let Ok(vrc) = vitaslop_gxp_shader::recompile_vertex(&gxp.vprog) {
            let mut regs = vitaslop_gxp_shader::interp::RegFile::with_lanes(512);
            for (k, c) in gxp.vert_sa.chunks_exact(4).enumerate() {
                if k < regs.sa.len() {
                    regs.sa[k] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
            }
            for a in &linked.vertex_bindings.attributes {
                if let Some(ga) = gxp.attributes.iter().find(|g| g.reg_index as u32 == a.base_lane) {
                    for c in 0..ga.components as usize {
                        let lane = a.base_lane as usize + c;
                        if lane < regs.pa.len() {
                            regs.pa[lane] = read_attr_component(&gxp.vertices, ga.offset as usize, ga.gxm_format, c);
                        }
                    }
                }
            }
            // The SECONDARY program runs before the primary and its whole purpose is to
            // leave values in SA registers the primary reads, so an interpretation that
            // skips it is reading uniforms the real module has already overwritten. Its
            // container literals go in too, as raw bit patterns, exactly as the module
            // emits them.
            if let Ok(program) = vitaslop_gxp_shader::Program::parse(&gxp.vprog) {
                for &(reg, value) in &program.literals {
                    if let Some(slot) = regs.sa.get_mut(reg as usize) {
                        *slot = f32::from_bits(value);
                    }
                }
                let secondary = vitaslop_gxp_shader::usse::decode_secondary_shader(&program);
                if let Err(e) = vitaslop_gxp_shader::interp::run(&secondary, &mut regs) {
                    report!("gxp interp: secondary program failed: {e}");
                }
            }
            // Where the WHOLE mesh lands, not just its first vertex. Vertex 0 of a large
            // world mesh is routinely behind the camera, so its `w<0` says nothing about
            // whether the draw covers any pixels - and "this draw is missing" is exactly
            // the question. Counting the vertices in front of the eye and taking the NDC
            // bounding box over them answers it outright.
            let saved_sa = regs.sa.clone();
            let stride = gxp.vertex_stride.max(1) as usize;
            let nverts = (gxp.vertices.len() / stride).min(4096);
            let (mut in_front, mut lo, mut hi) = (0usize, [f32::MAX; 2], [f32::MIN; 2]);
            for v in 0..nverts {
                let mut vregs = regs.clone();
                vregs.sa.clone_from(&saved_sa);
                for a in &linked.vertex_bindings.attributes {
                    if let Some(ga) = gxp.attributes.iter().find(|g| g.reg_index as u32 == a.base_lane) {
                        let base = v * stride + ga.offset as usize;
                        for c in 0..ga.components as usize {
                            let lane = a.base_lane as usize + c;
                            if lane < vregs.pa.len() {
                                vregs.pa[lane] = read_attr_component(&gxp.vertices, base, ga.gxm_format, c);
                            }
                        }
                    }
                }
                if vitaslop_gxp_shader::interp::run(&vrc.shader, &mut vregs).is_err() {
                    break;
                }
                let w = vregs.o[3];
                if w > 0.0 {
                    in_front += 1;
                    lo = [lo[0].min(vregs.o[0] / w), lo[1].min(vregs.o[1] / w)];
                    hi = [hi[0].max(vregs.o[0] / w), hi[1].max(vregs.o[1] / w)];
                }
            }
            report!(
                "gxp interp: key {key:x} {in_front}/{nverts} vertices have w>0, ndc bbox x[{:.3},{:.3}] y[{:.3},{:.3}]",
                lo[0], hi[0], lo[1], hi[1]
            );
            match vitaslop_gxp_shader::interp::run_watching_for_nan(&vrc.shader, &mut regs) {
                Ok(site) => {
                    let w = regs.o[3];
                    let ndc = if w.abs() > 1e-6 { [regs.o[0] / w, regs.o[1] / w, regs.o[2] / w] } else { [0.0; 3] };
                    report!("gxp interp: o={:?} ndc={:?} viewport(xo,xs,yo,ys,zo,zs)={:?}", &regs.o[0..4], ndc, gxp.viewport);
                    // A NaN clip position draws NOTHING, which looks exactly like a black
                    // shader; naming the instruction that produced it is the whole point.
                    if let Some(s) = site {
                        report!(
                            "gxp interp: FIRST non-finite value at instruction #{} {} -> {} channel {} = {}\n  sources: {}",
                            s.index, s.op, s.dest, s.channel, s.value, s.sources.join("  ")
                        );
                    }
                }
                Err(e) => report!("gxp interp: run failed: {e}"),
            }
        }

    // Through `KeySpec` so the knob takes `all` / a key list / `!<key list>`. The exclusion
    // form is the one that matters here: see `KeySpec::resolve`.
    let keycolor = {
        use std::sync::OnceLock;
        static SPEC: OnceLock<KeySpec> = OnceLock::new();
        SPEC.get_or_init(|| KeySpec::resolve("VITASLOP_GXP_KEYCOLOR")).wants(key).then_some(key)
    };
    let ret_spec = gxp_return_keys();
    let ret_restricted = !matches!(ret_spec, KeySpec::Off);
    let ret = !ret_restricted || ret_spec.wants(key);
    let wgsl = match inject_clip_fixup(&linked.wgsl, zfix, yflip, solid, keycolor, ret) {
        Some(w) => w,
        None => {
            crate::gpu::add_build_ms(&crate::gpu::PIPE_LINK_US, t_link.ms());
            report_fallback(
                key,
                "no `out.position` assignment to wrap with the clip fixup - refusing to \
                 render without the depth remap",
            );
            return None;
        }
    };
    // `VITASLOP_GXP_WGSL_DIR=<dir>`: write each pair's linked WGSL to `<key>.wgsl`.
    //
    // When a recompiled draw comes out wrong, the question is what the guest's shader was
    // actually translated INTO, and until now the only way to see that was to add a print
    // and rebuild. The translation is the artefact worth reading - it names the samplers,
    // the varyings and the clip position in one place.
    if let Ok(dir) = std::env::var("VITASLOP_GXP_WGSL_DIR") {
        let dir = std::path::Path::new(&dir);
        if let Err(e) = std::fs::create_dir_all(dir)
            .and_then(|()| std::fs::write(dir.join(format!("{key:016x}.wgsl")), &wgsl))
        {
            report_warn!("gxp: cannot write WGSL for pair {key:016x}: {e}");
        }
    }
    // DIAGNOSTIC (`VITASLOP_GXP_WGSL_OVERRIDE_DIR=<dir>`): when `<dir>/<key>.wgsl` exists,
    // compile THAT instead of the translation - edit a dumped module to return an
    // intermediate register and replay the frame, to ask which term of a shader goes wrong
    // without a rebuild. Never a fix: an overridden module is not the guest's program.
    // >>> PAIRS THAT SHARE A PROGRAM SHARE ONE COMPILED MODULE: the module cache is keyed on
    // the programs, not the pair key, so an override written for ONE key is silently
    // ignored if a sibling pair (same programs, different blend state) compiled first -
    // MEASURED: a hair pair's override changed nothing until the same file was written
    // under its two siblings' keys too. Override every key that shares the program, and do
    // not mix override and normal runs in one process.
    let wgsl = match crate::knobs::var("VITASLOP_GXP_WGSL_OVERRIDE_DIR") {
        Ok(dir) => match std::fs::read_to_string(std::path::Path::new(&dir).join(format!("{key:016x}.wgsl"))) {
            Ok(o) => {
                report_warn!("gxp pair {key:016x}: WGSL OVERRIDDEN from {dir} - this is not the guest's program");
                o
            }
            Err(_) => wgsl,
        },
        Err(_) => wgsl,
    };
    // The same override as TEXT (`VITASLOP_GXP_WGSL_OVERRIDE_TEXT=<hex key>\n<wgsl>`), for a
    // browser - a phone replay job has no filesystem, but its knobs are strings.
    let wgsl = match crate::knobs::var("VITASLOP_GXP_WGSL_OVERRIDE_TEXT") {
        Ok(t) => match t.split_once('\n') {
            Some((k, body)) if u64::from_str_radix(k.trim().trim_start_matches("0x"), 16).ok() == Some(key) => {
                report_warn!("gxp pair {key:016x}: WGSL OVERRIDDEN from VITASLOP_GXP_WGSL_OVERRIDE_TEXT - this is not the guest's program");
                body.to_string()
            }
            _ => wgsl,
        },
        Err(_) => wgsl,
    };
    // >>> THE JOIN BETWEEN A PAIR KEY AND ITS TWO BLOBS, once per pair.
    //
    // A pair KEY is what every attribution instrument here names - `VITASLOP_GXP_KEYCOLOR`,
    // `_KEYS`, `_EXCLUDE`, the WGSL dump. A BLOB CONTENT HASH is what the offline corpus
    // files a program under, and what `VITASLOP_GXP_CAPSULE` takes. Nothing printed the
    // mapping, so "the key that owns this black region" could not be turned into "capture
    // that draw" or "disassemble that program" without a rebuild - which is a step this
    // session paid for more than once.
    report!(
        "gxp pair {key:016x} = vertex {:016x} + fragment {:016x}",
        vitaslop_gxp_shader::container::Program::parse(&gxp.vprog).map(|p| p.hash).unwrap_or(0),
        vitaslop_gxp_shader::container::Program::parse(&gxp.fprog).map(|p| p.hash).unwrap_or(0),
    );
    // >>> TIMED, because "the first frame of a race costs 2.5 seconds" is not actionable
    // until it says WHICH half. A WGSL compile and a pipeline create are different costs
    // with different fixes: the compile depends on the shader pair alone and is now done
    // when `sceGxmShaderPatcherCreateFragmentProgram` names the pair (see
    // `GxmRenderer::precompile_pairs`), while the pipeline also needs the draw's blend,
    // depth, cull, format and sample count and cannot be. This lookup is what a precompiled
    // pair HITS; a miss compiles here exactly as it always did.
    // Everything above this line is ours; everything below is the driver's.
    crate::gpu::add_build_ms(&crate::gpu::PIPE_LINK_US, t_link.ms());
    let t_module = Stopwatch::start();
    let mkey = match keycolor {
        Some(_) => key,
        // A pair `VITASLOP_GXP_RETURN_KEYS` picked is a different module from the same blobs.
        None if ret_restricted && ret => key,
        // The dual-source module is a different module from the same two blobs - and so
        // are the raw-unit and raw-output variants (different bindings, a different entry).
        None => {
            GxpLive::module_key(&gxp.vprog, &gxp.fprog)
                ^ if dual_source { 0x5a5a_0000_0000_0001 } else { 0 }
                ^ raw_units.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ if raw64_output { 0x3c3c_0000_0000_0002 } else { 0 }
                // A per-sample destination read declares a multisampled texture and reads
                // `sample_index`: a different module from the same blobs.
                ^ if dest_ms { 0x7d7d_0000_0000_0010 } else { 0 }
                // >>> AND THE COLOUR MASK, because a program linked under a discarded
                // colour output is a DIFFERENT MODULE from the same two blobs: the
                // passthrough treatment that routes its varyings is off. Serving a
                // masked-off draw's module to an unmasked draw of the same pair would
                // paint the register file's initial state.
                ^ if colour_output_masked_off { 0x6969_0000_0000_0004 } else { 0 }
                // An alpha-splat module returns a different colour from the same blobs.
                ^ if alpha_single { 0x5151_0000_0000_0008 } else { 0 }
                // A fragment fed by the lane layout of the vertex it was PATCHED against
                // routes its varyings differently from the same two blobs linked by usage.
                ^ if gxp.fprog_patched_vprog.is_empty() {
                    0
                } else {
                    GxpLive::module_key(&gxp.fprog_patched_vprog, b"patched").rotate_left(17)
                }
                // >>> AND THE GUEST'S ATTRIBUTE FORMATS, because they decide a vertex
                // input's WGSL TYPE (`VertexAttribute::int_fetch`). Two guest layouts over
                // one program pair are two different modules, and serving the second from
                // the first's entry is a pipeline WebGPU refuses.
                ^ gxp_attr_interface_fold(&linked.vertex_bindings.attributes)
        }
    };
    // Which form the title's draws take - see `premodule_pairs`.
    if gxp.fprog_patched_vprog.is_empty() {
        crate::gpu::DRAWS_UNPATCHED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    } else {
        crate::gpu::DRAWS_PATCHED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let module = modules
        .entry(mkey)
        .or_insert_with(|| {
            // >>> LABELLED WITH THE PAIR, because the DEVICE's own error message is the
            // only diagnosis available on a phone.
            //
            // A device that rejects a module reports it as `[Invalid ShaderModule
            // "<label>"]`, and a constant label names nothing: a real report from an
            // Android PowerVR device read `[Invalid RenderPipeline "gxp"] is invalid due
            // to a previous error` thirty-two times, with no way to tell WHICH of the
            // title's pairs was bad or to look it up in the shader corpus. The module key
            // is the same hex a `gxp pair <key>` line prints, so the label joins the
            // device's complaint to the dump tooling.
            // The link's `diagnostic` directive (a derivative inside a loop) has to stay above
            // every declaration, and the clip fixup above prepends declarations.
            let wgsl = vitaslop_gxp_shader::wgsl::hoist_diagnostics(&wgsl);
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("gxp-linked:{mkey:016x}")),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            })
        })
        .clone();
    crate::gpu::add_build_ms(&crate::gpu::PIPE_MODULE_US, t_module.ms());

    // group0 vertex uniform, group1 fragment uniform, group2 samplers (empty where unused).
    //
    // Both are addressed by DYNAMIC OFFSET into one arena per pass, so a pass needs one
    // bind group per shader pair rather than one per draw - the same shape the
    // fixed-function path has always used for its per-draw uniform. `min_binding_size`
    // pins the shader-visible window to exactly this stage's SA block, so the arena's
    // 256-byte offset alignment cannot let a draw read into the next draw's uniforms.
    let uniform_entry = |vis: wgpu::ShaderStages, bytes: u64| wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: wgpu::BufferSize::new(bytes),
        },
        count: None,
    };
    let vsa_lanes = linked.vertex_bindings.sa_lane_count;
    let fsa_lanes = linked.fragment_bindings.sa_lane_count;
    let sa_bytes = |lanes: u32| (lanes.div_ceil(4) as u64) * 16;
    let (vsa_bytes, fsa_bytes) = (sa_bytes(vsa_lanes), sa_bytes(fsa_lanes));
    let mut g0_entries: Vec<wgpu::BindGroupLayoutEntry> =
        if vsa_lanes > 0 { vec![uniform_entry(wgpu::ShaderStages::VERTEX, vsa_bytes)] } else { vec![] };
    // The vertex stage's guest-memory window at group 0 binding 1, when its program
    // loads memory: the same dynamic-offset arena discipline as the SA blocks, with
    // `min_binding_size` pinning the shader-visible extent to exactly the declared
    // window (header vec4 included).
    //
    // >>> WINDOWS PAST 64 KiB BIND AS READ-ONLY STORAGE over the SAME arena - see
    // `vitaslop_gxp_shader::module::mem_window_storage`, the one answer the shader's own
    // declaration was emitted from. Everything that fits keeps its uniform binding.
    let mem_windows = linked.vertex_bindings.mem_windows.clone();
    let mem_bind_bytes =
        vitaslop_gxp_shader::module::mem_window_vec4_count(&mem_windows) as u64 * 16;
    let mem_storage = vitaslop_gxp_shader::module::mem_window_storage(&mem_windows);
    let fmem_windows = linked.fragment_bindings.mem_windows.clone();
    // A PER-DRAW window set binds as many bytes as each draw needs (see
    // `vitaslop_gxp_shader::module::MemWindow::per_draw` and `push_mem_windows_per_draw`): the
    // layout pins only its headers, and each draw's bind group names its own extent.
    let fmem_bind_bytes = if vitaslop_gxp_shader::module::mem_window_per_draw(&fmem_windows) {
        fmem_windows.len() as u64 * 16
    } else {
        vitaslop_gxp_shader::module::mem_window_vec4_count(&fmem_windows) as u64 * 16
    };
    let fmem_storage = vitaslop_gxp_shader::module::mem_window_storage(&fmem_windows);
    if let Some(why) = mem_storage_refusal(device, mem_storage.then_some(mem_bind_bytes), fmem_storage.then_some(fmem_bind_bytes)) {
        report_fallback(key, &why);
        return None;
    }
    if mem_storage || fmem_storage {
        note_mem_storage_pipeline();
    }
    let mem_entry = |vis: wgpu::ShaderStages, bytes: u64, storage: bool| wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty: if storage {
                wgpu::BufferBindingType::Storage { read_only: true }
            } else {
                wgpu::BufferBindingType::Uniform
            },
            has_dynamic_offset: true,
            min_binding_size: wgpu::BufferSize::new(bytes),
        },
        count: None,
    };
    if mem_bind_bytes > 0 {
        g0_entries.push(mem_entry(wgpu::ShaderStages::VERTEX, mem_bind_bytes, mem_storage));
    }
    let mut g1_entries: Vec<wgpu::BindGroupLayoutEntry> =
        if fsa_lanes > 0 { vec![uniform_entry(wgpu::ShaderStages::FRAGMENT, fsa_bytes)] } else { vec![] };
    // The FRAGMENT stage's own guest-memory window at group 1 binding 1 - the exact
    // mirror of the vertex one above, bound from the fragment uniform-buffer table.
    if fmem_bind_bytes > 0 {
        g1_entries.push(mem_entry(wgpu::ShaderStages::FRAGMENT, fmem_bind_bytes, fmem_storage));
    }
    let mut g2_entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();
    let mut samplers: Vec<(u8, SamplerDim)> = Vec::new();
    for (i, b) in linked.fragment_bindings.samplers.iter().enumerate() {
        // Mirrors `TexBinding::wgsl_type` exactly - the layout and the shader must agree.
        let dim = match (b.raw, b.coords >= 3, b.cube) {
            (true, ..) => SamplerDim::Raw,
            (false, true, true) => SamplerDim::Cube,
            (false, true, false) => SamplerDim::Three,
            _ if b.unit < 32 && f32_units & (1u64 << b.unit) != 0 => SamplerDim::TwoF32,
            _ => SamplerDim::Two,
        };
        // The binding plan already names the GXM texture unit the guest bound to: the SMP
        // sampler operand is resolved through the container's texture-control table when the
        // instruction stream is decoded, and a prefetched sample names its unit outright.
        let gxm_unit = b.unit as u32;
        g2_entries.extend(dim.layout_entries(i as u32 * 2, wgpu::ShaderStages::FRAGMENT));
        if debug {
            report!("gxp build: gxm texture unit {gxm_unit} (coords {}, {dim:?})", b.coords);
        }
        samplers.push((gxm_unit as u8, dim));
    }
    // group3 carries the scene depth range the injected clip fixup remaps through - one
    // vec4 the renderer refills per frame, so the pipeline stays cached by shader identity.
    // BOTH stages read it: the vertex stage to remap clip depth, the fragment stage to
    // reconstruct the guest's window POSITION (see `link::GXP_DEPTH_DECL`).
    // The VERTEX stage's samplers go in the SAME group, after the fragment ones: a device
    // guarantees only four bind groups and the other three are spoken for. A vertex program
    // that fetches a texture builds its geometry from it, so these are not decoration -
    // without them the draw has no mesh. Their unit numbering is independent of the fragment
    // stage's, which is why they keep separate names (`vt{u}`/`vs{u}`).
    let vsampler_base = g2_entries.len() as u32;
    let mut vertex_samplers: Vec<(u8, SamplerDim)> = Vec::new();
    for (i, b) in linked.vertex_bindings.samplers.iter().enumerate() {
        // A vertex stage is never linked with raw units (see `prepare`), so `raw` is false.
        let dim = match (b.coords >= 3, b.cube) {
            (true, true) => SamplerDim::Cube,
            (true, false) => SamplerDim::Three,
            _ if b.unit < 32 && f32_units & (1u64 << (32 + b.unit)) != 0 => SamplerDim::TwoF32,
            _ => SamplerDim::Two,
        };
        g2_entries.extend(dim.layout_entries(vsampler_base + i as u32 * 2, wgpu::ShaderStages::VERTEX));
        if debug {
            report!("gxp build: VERTEX gxm texture unit {} (coords {}, {dim:?})", b.unit, b.coords);
        }
        vertex_samplers.push((b.unit, dim));
    }
    // group3: the per-draw depth block, and - for a pair that blends for itself - the
    // DESTINATION colour it reads out of its output bank. The texture goes here because a
    // device guarantees only four bind groups and the other three are spoken for; group 3's
    // bind group is already rebuilt per draw state, so it is the one that can also carry a
    // view which changes WITHIN a pass.
    let mut g3_entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];
    if linked.reads_dest_color {
        g3_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            // A multisampled texture cannot be filterable-float; the program only loads it.
            // A raw 64-bit pass copies its `Rg32Uint` words (`module::dest_raw_init`).
            ty: wgpu::BindingType::Texture {
                sample_type: if raw64_output {
                    wgpu::TextureSampleType::Uint
                } else {
                    wgpu::TextureSampleType::Float { filterable: !dest_ms }
                },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: dest_ms,
            },
            count: None,
        });
    }
    // A FRAGMENT program that STORES to guest memory: its writable mirror (binding 2) and the
    // header block saying where each store window sits in it (binding 3) - see
    // `vitaslop_gxp_shader::module::store_window_decl` and `store_mirror`. A WRITABLE storage
    // buffer in the fragment stage is core WebGPU but not universal (WebGPU's compatibility
    // mode and some downlevel adapters lack it), so a device without it refuses the pair by
    // name rather than building a pipeline it rejects.
    let store_windows = linked.fragment_bindings.store_windows.clone();
    if !store_windows.is_empty() {
        let used = u32::from(fmem_storage) + 1;
        let why = if !crate::gpu::fragment_storage_ok() || crate::gpu::compat_mode() {
            Some("its fragment program STORES to guest memory, and this adapter cannot write a storage buffer from the fragment stage".to_string())
        } else if device.limits().max_storage_buffers_per_shader_stage < used {
            Some(format!(
                "its fragment program STORES to guest memory and needs {used} storage buffers in the fragment stage; this device offers {}",
                device.limits().max_storage_buffers_per_shader_stage
            ))
        } else {
            None
        };
        if let Some(why) = why {
            report_fallback(key, &why);
            return None;
        }
        g3_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        g3_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(store_windows.len() as u64 * 16),
            },
            count: None,
        });
    }
    let layouts = [
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("gxp-g0"), entries: &g0_entries }),
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("gxp-g1"), entries: &g1_entries }),
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("gxp-g2"), entries: &g2_entries }),
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gxp-g3"),
            // The pass depth block is per SCENE, not per draw: one cached bind group, no
            // dynamic offset.
            entries: &g3_entries,
        }),
    ];
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gxp-pl"),
        bind_group_layouts: &[Some(&layouts[0]), Some(&layouts[1]), Some(&layouts[2]), Some(&layouts[3])],
        immediate_size: 0,
    });

    let vbuffers: Vec<Option<wgpu::VertexBufferLayout>> = if wattrs.is_empty() {
        vec![]
    } else {
        vec![Some(wgpu::VertexBufferLayout { array_stride: array_stride as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wattrs })]
    };

    // The guest's own blend equation, from the `SceGxmBlendInfo` its fragment program was
    // created with. This is state, not a guess: GXM has no runtime blend setter, so the
    // old "opaque draws REPLACE, the rest src-over" reading was the renderer inventing a
    // mode. It got a title's whole world composite wrong - that draw does NOT blend, and
    // forcing src-over made it invisible the moment its shader output alpha 0.
    let guest_blend = gxm_blend_state(gxp.blend_state);
    // The colour channels the guest asked for - AND ONLY IF it left the fragment program
    // enabled. `sceGxmSetFrontFragmentProgramEnable(DISABLED)` makes the draw depth- and
    // stencil-only on hardware, whatever its shader computes; see
    // `GxpRecompile::fragment_program_enabled`.
    let write_mask = match gxp.fragment_program_enabled {
        true => gxm_color_mask(gxp.blend_state[0]),
        false => wgpu::ColorWrites::empty(),
    };
    // A colour mask of ZERO is a legal and meaningful GXM state - a depth-only prepass
    // writes no colour - but on screen it is indistinguishable from a draw we got wrong,
    // and it is the one pipeline setting that can make a perfectly recompiled, correctly
    // bound, correctly transformed draw leave no mark at all. Say it out loud, once.
    if write_mask.is_empty() && gxp.fragment_program_enabled {
        report_zero_color_mask(key);
    }
    // Depth comes from the guest's own captured state: `SceGxmDepthFunc` and the depth-write
    // enable, exactly as the hardware would apply them.
    //
    // It used to be a HEURISTIC - "an MVP-space draw that writes depth is opaque, everything
    // else is a 2D overlay" - decided in `render.rs` off the FIXED-FUNCTION reflection, i.e.
    // off whether we could recognise a model-view-projection matrix in the guest's uniform
    // buffer. A recompiled draw does not use that matrix at all, so the heuristic was
    // answering a question about a transform the pipeline never runs. On one title it
    // classified the engine's 2D primitive-render path as an overlay and gave a FULLSCREEN
    // black triangle `Always` + no depth write: it painted over the finished world every
    // frame, and no depth knob could move it because the draw was not being depth-tested at
    // all. Three different clip-depth remaps produced bit-identical black frames, which is
    // the fingerprint of depth state that is not the guest's.
    //
    // Switching this over was tried once before and MEASURED to be worse (a race's world
    // pass lost its track surface). That measurement was real, and it was against
    // `ZFix::Range`, which writes the scene's normalised `-1/w` - NOT the guest's depth. The
    // guest's own `LESS_EQUAL` cannot mean anything against a quantity the guest never
    // computed. The two changes only work together: see `ZFix::Off`, now the default, which
    // passes the guest's clip z through untouched so the buffer holds the guest's own
    // window depth.
    // >>> A BLEND THE SHADER PERFORMED ITSELF WINS OVER THE GUEST'S BLEND INFO, and the two
    // >>> can never both be present.
    //
    // GXM patches a blend into a fragment program from the `SceGxmBlendInfo` given at
    // creation; a program created with NONE can still blend, by reading its own output
    // registers - which are the ROP's destination colour - in ordinary ALU. When
    // `lower_dest_blend` recognises one of those it rewrites the program to emit only the
    // source term, and the equation it hands back IS this pipeline's blend. Applying the
    // guest's `REPLACE` instead would throw the destination away, which is the same black
    // frame the whole feature exists to fix.
    let guest_blend = match dest_blend_state(linked.dest_blend) {
        Some(b) => {
            report_lowered_dest_blend(key, linked.dest_blend);
            Some(b)
        }
        None => guest_blend,
    };
    // A DUAL-SOURCE lowering is the same override by a different route: the shader emits
    // `G` and `F` (see `module::dual_source_eligible`) and this blend is what makes them
    // `G + dst * F`. The guest's own blend info is NONE for such a program - it blended
    // in ALU precisely because it asked for no fixed-function blend.
    let guest_blend = if linked.dual_source {
        report_dual_source_blend(key, gxp.blend_state);
        Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::Src1,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::Src1Alpha,
                operation: wgpu::BlendOperation::Add,
            },
        })
    } else {
        guest_blend
    };
    let (mut blend, mut depth_write, mut depth_compare) =
        (guest_blend, gxp.depth_write, gxm_depth_func(gxp.depth_func));
    // >>> A SINGLE-CHANNEL ALPHA TARGET stores the fragment's alpha, which the link splats
    // into every channel (`LinkOptions::alpha_to_red`), and the one channel the target has
    // is written by the guest's ALPHA equation and only if its ALPHA write bit is set - see
    // `alpha_single_color_format`.
    let write_mask = if alpha_single && !raw64_output {
        if write_mask.contains(wgpu::ColorWrites::ALPHA) { wgpu::ColorWrites::ALL } else { wgpu::ColorWrites::empty() }
    } else {
        write_mask
    };
    if alpha_single && !raw64_output {
        blend = blend.map(alpha_single_blend);
    }
    if solid || noblend {
        // Diagnostic: REPLACE, so the fragment's colour reaches the target whatever its
        // alpha is. `solid` additionally replaces the shading and drops the depth test;
        // `noblend` changes ONLY the blend, which is what separates "this surface shades
        // black" from "this surface is composited away". A draw whose shader writes
        // alpha 0 under a src-alpha blend contributes exactly nothing, and the finished
        // frame cannot tell that apart from a fragment that computed black.
        blend = Some(wgpu::BlendState::REPLACE);
    }
    // A zero write mask is the same invisibility by a different route, so the diagnostic
    // has to lift it too or it only answers half the question.
    let write_mask = if solid || noblend { wgpu::ColorWrites::ALL } else { write_mask };
    if solid || nodepth {
        // Both diagnostics drop the depth test; `solid` additionally replaces the
        // shading, which is exactly the difference between them.
        depth_write = false;
        depth_compare = wgpu::CompareFunction::Always;
    }
    // An integer attachment cannot blend: the raw words are STORED, which is what the
    // hardware does with a 64-bit surface too. The blend a title asks for there is PATCHED
    // INTO its fragment program as ALU over the output registers (the destination), which this
    // link serves from a copy of the words - so applying the guest's blend state as well would
    // apply it twice. MEASURED on a title's light pass: every one of its five blended pairs
    // into an F16F16F16F16 surface reads the destination, and blending them again through a
    // half-float attachment doubled the destination under every additive sprite (hard-edged
    // rectangles over its title screen). Only a blended program that does NOT read the
    // destination has lost something; that one is reported, once.
    let blend = if raw64_output {
        if let Some(b) = blend.filter(|b| *b != wgpu::BlendState::REPLACE)
            && !fragment_reads_dest(&gxp.fprog)
        {
            report_raw64_blend_dropped(key, &b);
        }
        None
    } else {
        blend
    };
    report_gxp_pipeline_state(key, gxp, depth_write, depth_compare, blend, write_mask);
    // A DEPTH BIAS IS A POLYGON OFFSET, and WebGPU refuses one on a topology that has no
    // polygons: `depthBias must be 0 when using PrimitiveTopology::LineList`, and the same
    // for `depthBiasSlopeScale`. A refused pipeline drops every draw of its pair for the
    // rest of the run ([[vitaslop-one-refused-pipeline-blanks-the-frame]]), so this is not
    // a cosmetic guard - it is the same reason `cull_mode` is dropped just below, and it
    // was MISSED when line and point topologies stopped being dropped: this title carries
    // `sceGxmSetFrontDepthBias` on line draws, so two pairs were refused on a phone.
    // Nothing moves for a triangle: the bias is only zeroed where the device rejects it.
    let has_faces = gxm_topology_has_faces(gxp.primitive);
    let depth_bias = if has_faces { gxp.depth_bias } else { (0, 0) };
    // >>> THE DEPTH-ONLY FRAGMENT STAGE, for a draw whose colour is the destination.
    //
    // The colour this draw would write is the colour already there, proven from its own
    // uniform bytes (`draw_colour_is_noop`), so the whole fragment stage is dead work - and
    // on the title this exists for it is not small work: eighteen full-screen quads a frame
    // whose blend body is evaluated TWICE per fragment (the dual-source `G` and `F` arms)
    // over a texture sample and thirty-odd F16 operations.
    //
    // What cannot go is the DEPTH write, so the draw still has to rasterise. This keeps the
    // pair's own VERTEX module - the position and the depth are its output - and replaces
    // only the fragment stage, with an empty colour write mask and no blend.
    //
    // A trivial fragment stage rather than `fragment: None`: a pipeline with no fragment
    // state declares no colour targets, and a render pass that HAS a colour attachment is
    // then incompatible with it - the whole command buffer would be refused, which drops
    // every draw in the frame [[vitaslop-a-failed-submit-loses-the-whole-frame]]. One
    // target with `ColorWrites::empty()` keeps the pass compatible and writes nothing.
    //
    // The pipeline LAYOUT is the pair's own, unchanged, so the encoder binds exactly what it
    // binds for any other draw of this pair and the trivial stage simply uses none of it.
    let noop_module = depth_only.then(|| {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gxp:colour-noop"),
            source: wgpu::ShaderSource::Wgsl(
                "@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.0); }".into(),
            ),
        })
    });
    let t_pipe = Stopwatch::start();
    let make = || {
        create(device, &wgpu::RenderPipelineDescriptor {
            // Named by PAIR for the same reason the module above is - see there. This is
            // the label that appeared thirty-two times as a bare `"gxp"` in a device
            // failure report, naming nothing.
            label: Some(&format!("gxp:{key:016x}")),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_main"), buffers: &vbuffers, compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: noop_module.as_ref().unwrap_or(&module),
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: if noop_module.is_some() { None } else { blend },
                    write_mask: if noop_module.is_some() {
                        wgpu::ColorWrites::empty()
                    } else {
                        write_mask
                    },
                })],
                compilation_options: Default::default(),
            }),
            // The guest's own `SceGxmCullMode`, mapped by `gxm_cull_face` (which is where
            // the winding sense is derived and why). It used to be `None` unconditionally -
            // "draw both windings and rely on the depth test" - which is wrong twice over:
            // the software rasteriser HAS culled since it was written, so the two halves of
            // this renderer disagreed about which faces exist, and a back face that the
            // depth test happens to reject still costs its rasterisation. `solid` drops it
            // for the same reason it drops the depth test: that diagnostic exists to show
            // every fragment a pair produces.
            primitive: wgpu::PrimitiveState {
                topology: gxm_topology(gxp.primitive),
                cull_mode: if solid || !has_faces {
                    None
                } else {
                    gxm_cull_face(cull)
                },
                // CLAMP the depth instead of clipping the primitive, where the device
                // offers it: that is what PowerVR does, and it is why the clip fixup no
                // longer clamps in the vertex stage. See [`unclipped_depth_available`].
                unclipped_depth: crate::gpu::unclipped_depth_available(),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format(),
                depth_write_enabled: Some(depth_write),
                depth_compare: Some(depth_compare),
                // The guest's stencil test and ops - see `gxm_stencil_state`. The reference
                // is dynamic, set per draw where the draw is encoded.
                stencil: gxm_stencil_state(gxp),
                // GXM's `sceGxmSetFrontDepthBias(factor, units)` is the same polygon
                // offset wgpu spells `(slope_scale, constant)`: `factor` scales the
                // primitive's depth SLOPE and `units` the depth buffer's resolution
                // unit. The default `(0, 0)` is no bias, which is what all but the
                // decal draws carry - and it is what this was hardcoded to before the
                // state existed, so nothing that already rendered correctly moves.
                bias: wgpu::DepthBiasState {
                    constant: depth_bias.1,
                    slope_scale: depth_bias.0 as f32,
                    clamp: 0.0,
                },
            }),
            // MSAA, not alpha-to-coverage and not per-sample shading: the fragment stage
            // runs ONCE per pixel and every covered sample takes that result, which is
            // what `SceGxmMultisampleMode` asks for. `mask: !0` leaves coverage entirely
            // to the rasterizer, as the hardware does with no coverage override set.
            multisample: wgpu::MultisampleState {
                count: samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    };

    let pipeline = make();
    crate::gpu::add_build_ms(&crate::gpu::PIPE_CREATE_US, t_pipe.ms());
    if !keep_linked_modules() {
        modules.remove(&mkey);
    }
    Some(GxpPipelineOf {
        pipeline,
        layouts,
        vsa_lanes,
        fsa_lanes,
        mem_bind_bytes: mem_bind_bytes as u32,
        fmem_windows,
        fmem_bind_bytes: fmem_bind_bytes as u32,
        mem_windows,
        samplers,
        vertex_samplers,
        repack,
        packed_stride,
        packed_stride_native,
        passthrough,
        needs_dest: linked.reads_dest_color,
        store_windows,
    })
}

/// Whether a linked pair's shader module stays in `GxpLive::modules` once a pipeline has been
/// built from it. `VITASLOP_GXP_KEEP_MODULES=1` keeps it, `=0` releases it; unset releases on
/// the browser and keeps on native.
///
/// # >>> A HELD MODULE IS GPU-PROCESS MEMORY THE BROWSER CANNOT GIVE BACK
/// A WebGPU shader module keeps its parsed shader program alive in the GPU process for as long
/// as the page holds a handle to it; a pipeline built from it does not need that program again,
/// and the browser drops it once the last page-side handle goes. This cache held every module
/// for the life of the run, so it held every parsed program too.
///
/// MEASURED on the desktop browser, a first-person shooter's opening level (~940 pipelines from
/// ~500 modules), one build, this as the only difference: GPU process peak **3.97 GB -> 2.51
/// GB** private bytes, census `shaderModule live 501 -> 2`, speed 101% -> 102%. That title
/// crashed the phone's tab; the GPU process was the largest single holder.
///
/// What it costs: a later pipeline VARIANT of a pair already built (another format, sample
/// count or cull) compiles the module again. The text is linked and emitted before this cache
/// is consulted anyway, so only the browser's own compile repeats - and in the browser that is
/// the cheap half (`SHADER BUILDS` reads ~100 ms over a whole level). On NATIVE it is naga's
/// compile, the expensive half, and wgpu's pipelines hold their module regardless, so releasing
/// there buys nothing and costs compiles: native keeps them.
///
/// No picture in it: the same WGSL compiles to the same module.
pub(super) fn keep_linked_modules() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| match crate::knobs::var("VITASLOP_GXP_KEEP_MODULES").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        _ => !cfg!(target_arch = "wasm32"),
    })
}

/// Why this device cannot serve a pair whose guest-memory windows bind as STORAGE
/// (`vertex` / `fragment`: that stage's binding size when it is storage-bound), or `None`
/// when it can. Checked against what the device was actually CREATED with, so a refusal here
/// names a real device limit rather than surfacing later as an invalid bind group that loses
/// every submit naming it.
fn mem_storage_refusal(device: &wgpu::Device, vertex: Option<u64>, fragment: Option<u64>) -> Option<String> {
    if vertex.is_none() && fragment.is_none() {
        return None;
    }
    let l = device.limits();
    let biggest = vertex.unwrap_or(0).max(fragment.unwrap_or(0));
    let dynamic = u32::from(vertex.is_some()) + u32::from(fragment.is_some());
    if l.max_storage_buffers_per_shader_stage == 0 || l.max_dynamic_storage_buffers_per_pipeline_layout < dynamic {
        return Some(format!(
            "the guest-memory windows exceed a uniform binding and this device offers {} storage \
             buffers a stage and {} dynamic ones a layout - none to bind them as",
            l.max_storage_buffers_per_shader_stage, l.max_dynamic_storage_buffers_per_pipeline_layout
        ));
    }
    if biggest > l.max_storage_buffer_binding_size {
        return Some(format!(
            "the guest-memory windows need a {biggest}-byte storage binding and this device's \
             limit is {}",
            l.max_storage_buffer_binding_size
        ));
    }
    if vertex.is_some() && !crate::gpu::vertex_storage_ok() {
        return Some(
            "the VERTEX program's guest-memory windows exceed a uniform binding and this adapter \
             cannot read a storage buffer in the vertex stage"
                .to_string(),
        );
    }
    None
}
