//! Serialise ONE captured [`Draw`] to a file and read it back, so a draw can be re-rendered
//! offline instead of being reached by replaying the title.
//!
//! # Why this exists
//! Every `VITASLOP_GXP_*` diagnostic is read at STARTUP, so each question about what one
//! shader computes - "is this varying zero", "which term is the multiply by nothing" - costs a
//! full replay to the frame that draws it. On a retail title that is minutes per question, and
//! the questions come in ones: probe a register, look, probe the next. The prefix is not the
//! experiment, it is the tax on the experiment - the same observation the resident `session`
//! is built on, applied to the RENDERER instead of the scheduler.
//!
//! A capsule pays it once. Capture the draw during one replay; after that every probe,
//! substitution and knob combination is a second-long offline render of the same bytes.
//!
//! # What a capsule IS and, more importantly, what it is NOT
//! It is the [`Draw`] exactly as the capture built it: the guest's vertex and fragment program
//! blobs, both SA images, the memory windows, the vertex and index buffers, every bound
//! texture's DECODED texels, and the render/blend state. Re-rendering it runs the same
//! recompiler and the same pipeline build the live path runs, so a picture from a capsule is
//! evidence about the shader.
//!
//! It is NOT the frame. A capsule holds one draw, so it cannot answer anything about what was
//! drawn BEFORE it - no depth already in the buffer, no render-to-texture pass feeding a
//! sampler, no blending against neighbours. A surface that is wrong because something else
//! overwrote it looks fine in a capsule, and that is a difference the reader has to be told
//! about rather than left to discover, so [`Capsule::CAVEATS`] is printed by the replay tool
//! on every run.
//!
//! A bound texture is stored as the texels the capture DECODED, not as the guest's blocks: a
//! block-compressed original is re-encoded or expanded on the way to the GPU anyway
//! (`vitaslop-platform`'s upload path picks per adapter), and a capsule replays the decoded
//! seam on every adapter instead. Anything that turns on the exact block encoding has to be
//! asked of a live run.
//!
//! # Format
//! Little-endian, length-prefixed, no external dependency. Version is checked on read: an old
//! capsule is REFUSED rather than misread, because a capsule silently decoded through the wrong
//! field order would produce a wrong picture that looks like a shader bug.

use crate::capture::{BlendState, BoundTexture, Draw, FragmentMaterial, RenderState, VertexAttribute};
use std::io::{self, Read, Write};
use std::sync::Arc;

/// Magic + format version. Bump the version on ANY field-order change.
const MAGIC: &[u8; 8] = b"VSCAPS\x00\x01";

/// What a capsule cannot answer. Printed by the replay tool every time - a limitation nobody
/// reads is a limitation nobody applies.
pub const CAVEATS: &str = "\
a capsule is ONE draw, not the frame: there is no depth from earlier draws, no render-to-texture \
pass feeding a sampler, and no blending against neighbours. Textures replay from their DECODED \
texels, not the guest's blocks. Anything that turns on those needs a live run.";

/// A captured draw plus the target it was drawn into.
#[derive(Clone, Debug)]
pub struct Capsule {
    /// The draw itself, exactly as the capture built it.
    pub draw: Draw,
    /// Colour-target width/height the pass rasterised at.
    pub width: u32,
    pub height: u32,
    /// The clear colour the pass began with, so a replay starts from the same ground.
    pub clear: [u8; 4],
    /// The shader-pair key this draw was submitted under, for the record. A capsule that
    /// cannot say which pair it is gets mixed up with another one the moment there are two.
    pub key: u64,
    /// Display frame the draw was captured on, and its index within that scene.
    pub frame: u64,
    pub draw_index: u32,
    /// See [`CAVEATS`].
    pub note: String,
}

// --- primitive writers/readers ------------------------------------------------------

fn w_u8(o: &mut impl Write, v: u8) -> io::Result<()> {
    o.write_all(&[v])
}
fn w_u32(o: &mut impl Write, v: u32) -> io::Result<()> {
    o.write_all(&v.to_le_bytes())
}
fn w_i32(o: &mut impl Write, v: i32) -> io::Result<()> {
    o.write_all(&v.to_le_bytes())
}
fn w_u64(o: &mut impl Write, v: u64) -> io::Result<()> {
    o.write_all(&v.to_le_bytes())
}
fn w_f32(o: &mut impl Write, v: f32) -> io::Result<()> {
    o.write_all(&v.to_le_bytes())
}
fn w_bytes(o: &mut impl Write, v: &[u8]) -> io::Result<()> {
    w_u32(o, v.len() as u32)?;
    o.write_all(v)
}
fn w_f32s(o: &mut impl Write, v: &[f32]) -> io::Result<()> {
    for &f in v {
        w_f32(o, f)?;
    }
    Ok(())
}

fn r_u8(i: &mut impl Read) -> io::Result<u8> {
    let mut b = [0u8; 1];
    i.read_exact(&mut b)?;
    Ok(b[0])
}
fn r_u32(i: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    i.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn r_i32(i: &mut impl Read) -> io::Result<i32> {
    Ok(r_u32(i)? as i32)
}
fn r_u64(i: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    i.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn r_f32(i: &mut impl Read) -> io::Result<f32> {
    Ok(f32::from_bits(r_u32(i)?))
}
fn r_bytes(i: &mut impl Read) -> io::Result<Vec<u8>> {
    let n = r_u32(i)? as usize;
    let mut v = vec![0u8; n];
    i.read_exact(&mut v)?;
    Ok(v)
}
fn r_f32s<const N: usize>(i: &mut impl Read) -> io::Result<[f32; N]> {
    let mut a = [0f32; N];
    for s in a.iter_mut() {
        *s = r_f32(i)?;
    }
    Ok(a)
}

// --- component structs --------------------------------------------------------------

fn w_attr(o: &mut impl Write, a: &VertexAttribute) -> io::Result<()> {
    w_u32(o, a.stream_index as u32)?;
    w_u32(o, a.offset as u32)?;
    w_u8(o, a.format)?;
    w_u8(o, a.component_count)?;
    w_u32(o, a.reg_index as u32)
}
fn r_attr(i: &mut impl Read) -> io::Result<VertexAttribute> {
    Ok(VertexAttribute {
        stream_index: r_u32(i)? as u16,
        offset: r_u32(i)? as u16,
        format: r_u8(i)?,
        component_count: r_u8(i)?,
        reg_index: r_u32(i)? as u16,
    })
}

fn w_tex(o: &mut impl Write, t: &BoundTexture) -> io::Result<()> {
    for v in [
        t.unit, t.base_format, t.swizzle, t.tex_type, t.width, t.height, t.stride, t.faces,
        t.face_bytes, t.levels, t.data_addr,
    ] {
        w_u32(o, v)?;
    }
    w_bytes(o, &t.pixels)?;
    for v in [t.u_addr_mode, t.v_addr_mode, t.lod_bias, t.min_filter, t.mag_filter, t.mip_filter, t.gamma] {
        w_u32(o, v)?;
    }
    Ok(())
}
fn r_tex(i: &mut impl Read) -> io::Result<BoundTexture> {
    let (unit, base_format, swizzle, tex_type) = (r_u32(i)?, r_u32(i)?, r_u32(i)?, r_u32(i)?);
    let (width, height, stride, faces) = (r_u32(i)?, r_u32(i)?, r_u32(i)?, r_u32(i)?);
    let (face_bytes, levels, data_addr) = (r_u32(i)?, r_u32(i)?, r_u32(i)?);
    let pixels: Arc<[u8]> = Arc::from(r_bytes(i)?);
    Ok(BoundTexture {
        unit,
        // Replayed from a capsule: a buffer this process just built, so it is minted like any
        // other. A capsule replays ONE draw, so nothing downstream is comparing it to another.
        pixels_id: crate::capture::next_pixels_id(),
        base_format,
        swizzle,
        tex_type,
        width,
        height,
        stride,
        faces,
        face_bytes,
        levels,
        data_addr,
        pixels,
        u_addr_mode: r_u32(i)?,
        v_addr_mode: r_u32(i)?,
        lod_bias: r_u32(i)?,
        min_filter: r_u32(i)?,
        mag_filter: r_u32(i)?,
        mip_filter: r_u32(i)?,
        gamma: r_u32(i)?,
    })
}

fn w_state(o: &mut impl Write, s: &RenderState) -> io::Result<()> {
    for v in [s.cull_mode, s.two_sided, s.front_depth_func] {
        w_u32(o, v)?;
    }
    w_i32(o, s.front_depth_bias_factor)?;
    w_i32(o, s.front_depth_bias_units)?;
    for v in [
        s.back_depth_func,
        s.front_depth_write,
        s.back_depth_write,
        s.front_fragment_program_enable,
        s.back_fragment_program_enable,
        s.front_polygon_mode,
        s.back_polygon_mode,
        s.front_point_line_width,
        s.front_stencil_ref,
        s.front_stencil_func,
        s.front_stencil_op_fail,
        s.front_stencil_op_depth_fail,
        s.front_stencil_op_depth_pass,
        s.front_stencil_compare_mask,
        s.front_stencil_write_mask,
        s.back_stencil_func,
        s.back_stencil_op_fail,
        s.back_stencil_op_depth_fail,
        s.back_stencil_op_depth_pass,
        s.back_stencil_compare_mask,
        s.back_stencil_write_mask,
        s.viewport_enable,
    ] {
        w_u32(o, v)?;
    }
    w_f32s(o, &s.viewport)?;
    w_u32(o, s.region_clip_mode)?;
    for v in s.region_clip {
        w_u32(o, v)?;
    }
    for v in [s.front_visibility_test_enable, s.front_visibility_test_index, s.front_visibility_test_op] {
        w_u32(o, v)?;
    }
    Ok(())
}
fn r_state(i: &mut impl Read) -> io::Result<RenderState> {
    let mut s = RenderState::default();
    s.cull_mode = r_u32(i)?;
    s.two_sided = r_u32(i)?;
    s.front_depth_func = r_u32(i)?;
    s.front_depth_bias_factor = r_i32(i)?;
    s.front_depth_bias_units = r_i32(i)?;
    s.back_depth_func = r_u32(i)?;
    s.front_depth_write = r_u32(i)?;
    s.back_depth_write = r_u32(i)?;
    s.front_fragment_program_enable = r_u32(i)?;
    s.back_fragment_program_enable = r_u32(i)?;
    s.front_polygon_mode = r_u32(i)?;
    s.back_polygon_mode = r_u32(i)?;
    s.front_point_line_width = r_u32(i)?;
    s.front_stencil_ref = r_u32(i)?;
    s.front_stencil_func = r_u32(i)?;
    s.front_stencil_op_fail = r_u32(i)?;
    s.front_stencil_op_depth_fail = r_u32(i)?;
    s.front_stencil_op_depth_pass = r_u32(i)?;
    s.front_stencil_compare_mask = r_u32(i)?;
    s.front_stencil_write_mask = r_u32(i)?;
    s.back_stencil_func = r_u32(i)?;
    s.back_stencil_op_fail = r_u32(i)?;
    s.back_stencil_op_depth_fail = r_u32(i)?;
    s.back_stencil_op_depth_pass = r_u32(i)?;
    s.back_stencil_compare_mask = r_u32(i)?;
    s.back_stencil_write_mask = r_u32(i)?;
    s.viewport_enable = r_u32(i)?;
    s.viewport = r_f32s::<6>(i)?;
    s.region_clip_mode = r_u32(i)?;
    for k in 0..4 {
        s.region_clip[k] = r_u32(i)?;
    }
    s.front_visibility_test_enable = r_u32(i)?;
    s.front_visibility_test_index = r_u32(i)?;
    s.front_visibility_test_op = r_u32(i)?;
    Ok(s)
}

// --- the capsule ---------------------------------------------------------------------

impl Capsule {
    /// See the module docs. Kept on the type too so a caller that only has a `Capsule` in scope
    /// still finds it.
    pub const CAVEATS: &'static str = CAVEATS;

    /// Encode to a byte stream.
    pub fn write(&self, o: &mut impl Write) -> io::Result<()> {
        o.write_all(MAGIC)?;
        w_u64(o, self.key)?;
        w_u64(o, self.frame)?;
        w_u32(o, self.draw_index)?;
        w_u32(o, self.width)?;
        w_u32(o, self.height)?;
        o.write_all(&self.clear)?;
        w_bytes(o, self.note.as_bytes())?;

        let d = &self.draw;
        w_u32(o, d.primitive)?;
        w_u32(o, d.index_format)?;
        w_u32(o, d.index_count)?;
        w_bytes(o, &d.vertices)?;
        w_u32(o, d.vertex_stride)?;
        w_u32(o, d.attributes.len() as u32)?;
        for a in d.attributes.iter() {
            w_attr(o, a)?;
        }
        w_bytes(o, &d.indices)?;
        w_u32(o, d.uniforms.len() as u32)?;
        w_f32s(o, &d.uniforms)?;
        w_u32(o, d.textures.len() as u32)?;
        for t in d.textures.iter() {
            w_tex(o, t)?;
        }
        w_u32(o, d.vertex_textures.len() as u32)?;
        for t in d.vertex_textures.iter() {
            w_tex(o, t)?;
        }
        w_state(o, &d.render_state)?;
        for v in [
            d.blend.color_mask,
            d.blend.color_func,
            d.blend.alpha_func,
            d.blend.color_src,
            d.blend.color_dst,
            d.blend.alpha_src,
            d.blend.alpha_dst,
        ] {
            w_u8(o, v)?;
        }
        w_u32(o, d.fragment_program_header)?;
        w_f32(o, d.exposure)?;
        w_f32s(o, &d.material.tint)?;
        w_f32s(o, &d.material.light_dir)?;
        w_f32s(o, &d.material.light_col)?;
        w_u8(o, d.material.has_light as u8)?;
        w_f32s(o, &d.material.ambient)?;
        w_f32s(o, &d.world)?;
        w_bytes(o, &d.vprog)?;
        w_bytes(o, &d.fprog)?;
        w_bytes(o, &d.vert_sa)?;
        w_bytes(o, &d.frag_sa)?;
        w_u32(o, d.frag_sa_addr)?;
        w_u32(o, d.mem_windows.len() as u32)?;
        for (addr, bytes) in &d.mem_windows {
            w_u32(o, *addr)?;
            w_bytes(o, bytes)?;
        }
        w_u8(o, d.shader_expanded as u8)?;
        Ok(())
    }

    /// Decode from a byte stream. A capsule whose magic or version does not match is REFUSED:
    /// decoding one through the wrong field order produces a wrong picture that reads exactly
    /// like a shader bug, which is the most expensive kind of wrong answer this can give.
    pub fn read(i: &mut impl Read) -> io::Result<Capsule> {
        let mut magic = [0u8; 8];
        i.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "not a capsule of this version (magic {magic:?}, expected {MAGIC:?}) - \
                     re-capture it rather than reading it through a different field order"
                ),
            ));
        }
        let key = r_u64(i)?;
        let frame = r_u64(i)?;
        let draw_index = r_u32(i)?;
        let width = r_u32(i)?;
        let height = r_u32(i)?;
        let mut clear = [0u8; 4];
        i.read_exact(&mut clear)?;
        let note = String::from_utf8_lossy(&r_bytes(i)?).into_owned();

        let primitive = r_u32(i)?;
        let index_format = r_u32(i)?;
        let index_count = r_u32(i)?;
        let vertices: Arc<[u8]> = Arc::from(r_bytes(i)?);
        let vertex_stride = r_u32(i)?;
        let n = r_u32(i)? as usize;
        let mut attributes = Vec::with_capacity(n);
        for _ in 0..n {
            attributes.push(r_attr(i)?);
        }
        let indices: Arc<[u8]> = Arc::from(r_bytes(i)?);
        let n = r_u32(i)? as usize;
        let mut uniforms = Vec::with_capacity(n);
        for _ in 0..n {
            uniforms.push(r_f32(i)?);
        }
        let n = r_u32(i)? as usize;
        let mut textures = Vec::with_capacity(n);
        for _ in 0..n {
            textures.push(r_tex(i)?);
        }
        let n = r_u32(i)? as usize;
        let mut vertex_textures = Vec::with_capacity(n);
        for _ in 0..n {
            vertex_textures.push(r_tex(i)?);
        }
        let render_state = Arc::new(r_state(i)?);
        let blend = BlendState {
            color_mask: r_u8(i)?,
            color_func: r_u8(i)?,
            alpha_func: r_u8(i)?,
            color_src: r_u8(i)?,
            color_dst: r_u8(i)?,
            alpha_src: r_u8(i)?,
            alpha_dst: r_u8(i)?,
        };
        let fragment_program_header = r_u32(i)?;
        let exposure = r_f32(i)?;
        let material = FragmentMaterial {
            tint: r_f32s::<3>(i)?,
            light_dir: r_f32s::<3>(i)?,
            light_col: r_f32s::<3>(i)?,
            has_light: r_u8(i)? != 0,
            ambient: r_f32s::<3>(i)?,
        };
        let world = r_f32s::<16>(i)?;
        let vprog: Arc<[u8]> = Arc::from(r_bytes(i)?);
        let fprog: Arc<[u8]> = Arc::from(r_bytes(i)?);
        let vert_sa: Arc<[u8]> = Arc::from(r_bytes(i)?);
        let frag_sa: Arc<[u8]> = Arc::from(r_bytes(i)?);
        let frag_sa_addr = r_u32(i)?;
        let n = r_u32(i)? as usize;
        let mut mem_windows = Vec::with_capacity(n);
        for _ in 0..n {
            let addr = r_u32(i)?;
            mem_windows.push((addr, r_bytes(i)?));
        }
        let shader_expanded = r_u8(i)? != 0;

        Ok(Capsule {
            draw: Draw {
                primitive,
                index_format,
                index_count,
                vertices,
                vertex_stride,
                attributes: Arc::from(attributes),
                indices,
                uniforms,
                textures: Arc::from(textures),
                vertex_textures: Arc::from(vertex_textures),
                render_state,
                blend,
                fragment_program_header,
                exposure,
                material,
                world,
                vprog,
                fprog,
                vert_sa,
                frag_sa,
                frag_sa_addr,
                mem_windows,
                shader_expanded,
            },
            width,
            height,
            clear,
            key,
            frame,
            draw_index,
            note,
        })
    }

    /// Write to a path.
    pub fn save(&self, path: &std::path::Path) -> io::Result<()> {
        let mut buf = Vec::new();
        self.write(&mut buf)?;
        std::fs::write(path, buf)
    }

    /// Read from a path.
    pub fn load(path: &std::path::Path) -> io::Result<Capsule> {
        let bytes = std::fs::read(path)?;
        Capsule::read(&mut &bytes[..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A capsule must ROUND-TRIP exactly. This is the whole guarantee: a replayed draw that
    /// differs from the captured one in any field is a wrong picture attributed to the shader.
    #[test]
    fn capsule_round_trips_every_field() {
        let attrs = vec![
            VertexAttribute { stream_index: 0, offset: 0, format: 9, component_count: 4, reg_index: 0 },
            VertexAttribute { stream_index: 0, offset: 52, format: 4, component_count: 4, reg_index: 8 },
        ];
        let tex = BoundTexture {
            // A fixture: a DISTINCT buffer, so a distinct identity - two fixtures sharing
            // one id would collide in every cache keyed on it.
            pixels_id: crate::capture::next_pixels_id(),
            unit: 1,
            base_format: 0x85,
            swizzle: 3,
            tex_type: 3,
            width: 4,
            height: 2,
            stride: 16,
            faces: 1,
            face_bytes: 32,
            levels: 2,
            data_addr: 0x8bb1_ac10,
            pixels: Arc::from(vec![1u8, 2, 3, 4, 5, 6, 7, 8]),
            u_addr_mode: 1,
            v_addr_mode: 2,
            lod_bias: 3,
            min_filter: 1,
            mag_filter: 0,
            mip_filter: 1,
            gamma: 1,
        };
        let mut state = RenderState::default();
        state.cull_mode = 2;
        state.front_depth_func = 3;
        state.front_depth_bias_factor = -4;
        state.front_depth_bias_units = 7;
        state.viewport = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        state.region_clip_mode = 0xC000_0000;
        state.region_clip = [9, 8, 7, 6];
        state.front_visibility_test_index = 5;
        let draw = Draw {
            primitive: 1,
            index_format: 0,
            index_count: 3,
            vertices: Arc::from(vec![9u8; 24]),
            vertex_stride: 12,
            attributes: Arc::from(attrs),
            indices: Arc::from(vec![0u8, 0, 1, 0, 2, 0]),
            uniforms: vec![0.5, -1.5],
            textures: Arc::from(vec![tex.clone()]),
            vertex_textures: Arc::from(vec![tex]),
            render_state: Arc::new(state),
            blend: BlendState {
                color_mask: 0xf,
                color_func: 1,
                alpha_func: 2,
                color_src: 3,
                color_dst: 4,
                alpha_src: 5,
                alpha_dst: 6,
            },
            fragment_program_header: 0x8100_0000,
            exposure: 2.25,
            material: FragmentMaterial {
                tint: [0.1, 0.2, 0.3],
                light_dir: [0.4, 0.5, 0.6],
                light_col: [0.7, 0.8, 0.9],
                has_light: true,
                ambient: [1.0, 1.1, 1.2],
            },
            world: [0.0; 16],
            vprog: Arc::from(vec![0xAAu8; 5]),
            fprog: Arc::from(vec![0xBBu8; 6]),
            vert_sa: Arc::from(vec![0xCCu8; 7]),
            frag_sa: Arc::from(vec![0xDDu8; 8]),
            frag_sa_addr: 0x882c_aa80,
            mem_windows: vec![(0x882c_9780, vec![1, 2, 3, 4]), (0x8e1d_2fb0, vec![5, 6])],
            shader_expanded: true,
        };
        let c = Capsule {
            draw,
            width: 960,
            height: 544,
            clear: [16, 16, 24, 255],
            key: 0xae49_3765_df2c_56b5,
            frame: 4600,
            draw_index: 42,
            note: "world pair".into(),
        };

        let mut buf = Vec::new();
        c.write(&mut buf).expect("write");
        let mut back = Capsule::read(&mut &buf[..]).expect("read");

        // >>> `pixels_id` IS DELIBERATELY NOT SERIALISED, so it is normalised before the
        // whole-struct compare below.
        //
        // It is a PROCESS-LOCAL identity, minted so that two different buffers can never
        // collide in a cache keyed on it (see `capture::BoundTexture::pixels_id`). Writing it
        // into the file and reading it back would import numbers minted by another process into
        // this one's sequence, where they can collide with freshly minted ones - which is the
        // exact defect the field exists to remove. The reader mints instead, and a capsule
        // replays ONE draw, so nothing is comparing two of them.
        let same_ids = |read: &Arc<[BoundTexture]>, want: &Arc<[BoundTexture]>| -> Arc<[BoundTexture]> {
            read.iter()
                .zip(want.iter())
                .map(|(a, b)| {
                    let mut a = a.clone();
                    a.pixels_id = b.pixels_id;
                    a
                })
                .collect()
        };
        back.draw.textures = same_ids(&back.draw.textures, &c.draw.textures);
        back.draw.vertex_textures = same_ids(&back.draw.vertex_textures, &c.draw.vertex_textures);

        assert_eq!(back.key, c.key);
        assert_eq!(back.frame, c.frame);
        assert_eq!(back.draw_index, c.draw_index);
        assert_eq!((back.width, back.height), (c.width, c.height));
        assert_eq!(back.clear, c.clear);
        assert_eq!(back.note, c.note);
        // `Draw` derives PartialEq, so this compares EVERY field - including any added later,
        // which is exactly the failure this test exists to catch.
        assert_eq!(back.draw, c.draw);
    }

    /// A capsule from a different format version must be refused, not misread.
    #[test]
    fn a_capsule_with_the_wrong_magic_is_refused() {
        let mut bytes = vec![0u8; 64];
        bytes[..8].copy_from_slice(b"VSCAPS\x00\x00");
        let e = Capsule::read(&mut &bytes[..]).expect_err("must refuse");
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
    }
}

// --- capture -------------------------------------------------------------------------

/// Diagnostic (`VITASLOP_GXP_CAPSULE=<vprog-hash>[,<vprog-hash>]:<dir>[:N]`): write the first
/// `N` (default 1) distinct submissions of each named VERTEX PROGRAM out as capsules.
///
/// The selector is the vertex program's CONTENT hash - the same value
/// `gxp pair <key>: vprog hash ...` prints and the same one the offline corpus names its blobs
/// by - deliberately NOT the pipeline pair key. That key is a pointer-identity cache built at
/// draw time, so it is not stable between runs and cannot be looked up from a previous log; a
/// content hash is the same number in every run and in the corpus.
///
/// Distinctness is on the bytes that make a draw a different EXPERIMENT: both SA images, the
/// memory windows and the vertex buffer. Writing only the literally-first submission of a pair
/// is how a capture ends up describing a draw nobody was asking about - one pair is submitted
/// many times a frame with different uniforms, which is what a per-draw uniform buffer is for.
/// Diagnostic (`VITASLOP_GXP_CAPSULE_MIN_INDICES=<n>`): only capture draws with at least this
/// many indices. Default 0, which captures whatever comes first.
///
/// Without it a capture takes the FIRST submissions of a program, and the first submissions of
/// a world material are whatever small object the title drew first - on one retail title, four
/// front-end props of 90 to 190 triangles, none of them the surface being investigated, and all
/// four replaying to a picture that says nothing. The draw one actually wants is usually the
/// LARGEST: a terrain mesh is tens of thousands of indices where a prop is hundreds, so a single
/// threshold separates them cleanly and needs no frame number to be known in advance.
fn min_indices() -> u32 {
    use std::sync::OnceLock;
    static MIN: OnceLock<u32> = OnceLock::new();
    *MIN.get_or_init(|| {
        vitaslop_platform::knobs::var("VITASLOP_GXP_CAPSULE_MIN_INDICES")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    })
}

/// Diagnostic (`VITASLOP_GXP_CAPSULE_SKIP=<n>`): ignore the first `n` matching submissions
/// before capturing anything. Default 0.
///
/// This is how a capture reaches a draw LATER in the run without a frame number having to be
/// plumbed into the scene builder. Submissions of one program arrive in time order, so skipping
/// past the count a title's front end makes lands the capture in gameplay - and unlike a size
/// threshold it works when the surface of interest is drawn as many small meshes rather than
/// one big one, which is the usual shape for terrain.
fn skip_count() -> usize {
    use std::sync::OnceLock;
    static SKIP: OnceLock<usize> = OnceLock::new();
    *SKIP.get_or_init(|| {
        vitaslop_platform::knobs::var("VITASLOP_GXP_CAPSULE_SKIP")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    })
}

pub fn maybe_capture(d: &Draw) {
    use std::collections::HashSet;
    use std::hash::{Hash, Hasher};
    use std::sync::{Mutex, OnceLock};

    // Resolved once. The off path must cost nothing: this runs per draw, thousands of times a
    // frame, and an `env::var` plus a parse there is real time on a diagnostic nobody enabled.
    struct Spec {
        hashes: Vec<u64>,
        dir: std::path::PathBuf,
        limit: usize,
    }
    static SPEC: OnceLock<Option<Spec>> = OnceLock::new();
    let spec = SPEC.get_or_init(|| {
        let raw = vitaslop_platform::knobs::var("VITASLOP_GXP_CAPSULE").ok()?;
        // `<hashes>:<dir>[:N]`, where the dir may itself contain a drive-letter colon.
        let (hashes, rest) = raw.split_once(':')?;
        let (dir, limit) = match rest.rsplit_once(':') {
            Some((d, n)) if n.chars().all(|c| c.is_ascii_digit()) && !n.is_empty() => {
                (d, n.parse().unwrap_or(1))
            }
            _ => (rest, 1usize),
        };
        let hashes: Vec<u64> = hashes
            .split(',')
            .filter_map(|h| u64::from_str_radix(h.trim().trim_start_matches("0x"), 16).ok())
            .collect();
        if hashes.is_empty() {
            return None;
        }
        let _ = std::fs::create_dir_all(dir);
        Some(Spec { hashes, dir: std::path::PathBuf::from(dir), limit })
    });
    let Some(spec) = spec else { return };

    let vhash = match vitaslop_gxp_shader::Program::parse(&d.vprog) {
        Ok(p) => p.hash,
        Err(_) => return,
    };
    if !spec.hashes.contains(&vhash) {
        return;
    }

    // A capture that writes nothing must say what it DID see, or "no capsules" is
    // indistinguishable from "the program never drew" - and the two need opposite next steps.
    // Tallied per program: how many submissions matched the hash, and the largest index count
    // among them, which is exactly what `_MIN_INDICES` has to be set below.
    {
        use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
        static SEEN_ANY: AtomicU64 = AtomicU64::new(0);
        static MAX_IDX: AtomicU32 = AtomicU32::new(0);
        let n = SEEN_ANY.fetch_add(1, Ordering::Relaxed) + 1;
        MAX_IDX.fetch_max(d.index_count, Ordering::Relaxed);
        // Every power-of-two submission, so a long run reports a handful of times rather than
        // thousands and still reports early enough to be useful.
        if n.is_power_of_two() {
            tracing::warn!(
                target: "vitaslop::gxm",
                "gxp capsule: {n} submissions of a named program so far, largest {} indices                  (set VITASLOP_GXP_CAPSULE_MIN_INDICES at or below that, and _SKIP below {n})",
                MAX_IDX.load(Ordering::Relaxed)
            );
        }
    }


    if d.index_count < min_indices() {
        return;
    }

    let mut h = std::collections::hash_map::DefaultHasher::new();
    d.vert_sa.hash(&mut h);
    d.frag_sa.hash(&mut h);
    d.mem_windows.hash(&mut h);
    d.vertices.hash(&mut h);
    let inputs = h.finish();

    static SEEN: OnceLock<Mutex<HashSet<(u64, u64)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.insert((vhash, inputs)) {
        return;
    }
    let n = seen.iter().filter(|(v, _)| *v == vhash).count();
    let skip = skip_count();
    if n <= skip || n > skip + spec.limit {
        return;
    }
    let n = n - skip;

    // The pass extent, from the guest's own viewport: GXM stores half-extents as the scale
    // terms, so the rasterised size is twice their magnitude. Falling back to the panel when a
    // draw carries no viewport is a GUESS and is recorded as one in the note.
    let vp = d.render_state.viewport;
    let (w, h_) = ((vp[1].abs() * 2.0) as u32, (vp[3].abs() * 2.0) as u32);
    let guessed = w == 0 || h_ == 0;
    let (w, h_) = if guessed { (960, 544) } else { (w, h_) };

    let path = spec.dir.join(format!("v{vhash:016x}-{n}.capsule"));
    let c = Capsule {
        draw: d.clone(),
        width: w,
        height: h_,
        clear: [16, 16, 24, 255],
        key: vhash,
        frame: 0,
        draw_index: n as u32,
        note: format!(
            "vprog {vhash:016x} submission {n}; extent {w}x{h_}{}",
            if guessed { " (GUESSED - this draw carried no viewport)" } else { " (from the viewport)" }
        ),
    };
    match c.save(&path) {
        Ok(()) => tracing::warn!(
            target: "vitaslop::gxm",
            "gxp capsule: wrote {} ({} vertex bytes, {} indices, {} textures) - replay it offline instead of \
             replaying the title. {}",
            path.display(),
            d.vertices.len(),
            d.index_count,
            d.textures.len(),
            CAVEATS
        ),
        // A diagnostic that cannot write must SAY so: a silent failure here reads as "the draw
        // never happened", which sends the reader to look for it in the wrong place entirely.
        Err(e) => tracing::warn!(
            target: "vitaslop::gxm",
            "gxp capsule: could NOT write {}: {e}", path.display()
        ),
    }
}
