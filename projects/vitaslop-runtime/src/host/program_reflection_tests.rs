//! The reflected extent of a program's default uniform buffer, in 32-bit registers.
//!
//! This is the size the capture reads out of the guest's reserved buffer, and the
//! buffer is a RECYCLED per-scene ring - so an over-read does not return zeros, it
//! returns a neighbouring draw's uniforms. That is a wrong answer wearing the shape
//! of a real one, which is why the width table is tested per type rather than on the
//! one program that exposed it.
use super::*;

/// Build a `SceGxmProgram` container just complete enough for
/// [`reflect_program_uncached`]: the parameter count at +0x24, the table offset at
/// +0x28, and one 16-byte record per parameter.
///
/// A record is `{ +0 name offset (relative to the record), +4 packed type word,
/// +8 array size, +0xc resource index }`, and the packed word is
/// `category | type << 4 | components << 8`.
fn program_with_uniforms(params: &[(u32, u32, u32, u32)]) -> Vec<u8> {
    const TABLE: u32 = 0x40;
    let mut bytes = vec![0u8; (TABLE as usize) + params.len() * 16 + 64];
    let put = |b: &mut Vec<u8>, at: u32, v: u32| {
        b[at as usize..at as usize + 4].copy_from_slice(&v.to_le_bytes());
    };
    put(&mut bytes, 0x24, params.len() as u32);
    // The table offset is stored RELATIVE to the field that holds it.
    put(&mut bytes, 0x28, TABLE - 0x28);
    for (i, &(ptype, comp, array, res)) in params.iter().enumerate() {
        let p = TABLE + i as u32 * 16;
        // Name offset 0 points the name reader at the record itself, which is a
        // zero byte - an empty name. None of the name-matched fields are under
        // test here, and an empty name cannot accidentally match one.
        put(&mut bytes, p, 0);
        put(&mut bytes, p + 4, 1 | (ptype << 4) | (comp << 8));
        put(&mut bytes, p + 8, array);
        put(&mut bytes, p + 0xc, res);
    }
    bytes
}

fn uniform_bytes(params: &[(u32, u32, u32, u32)]) -> u32 {
    let mut bytes = program_with_uniforms(params);
    let mut regs = [0u32; REG_COUNT];
    let mut vfp = [0u32; VFP_ARG_COUNT];
    let mut mem = SliceMemory(&mut bytes);
    let ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
    reflect_program_uncached(&ctx, 0).uniform_size_bytes
}

#[test]
fn f32_parameters_take_one_register_per_component() {
    // A 4x4 matrix is declared components 4, array 4 - sixteen registers.
    assert_eq!(uniform_bytes(&[(0, 4, 4, 0)]), 64);
    // ...and one starting at register 16 ends at 32.
    assert_eq!(uniform_bytes(&[(0, 4, 4, 0), (0, 4, 4, 16)]), 128);
}

#[test]
fn f16_parameters_pack_two_components_per_register() {
    // THE CASE THAT WAS WRONG. This title's tone-map fragment program declares
    // `exposure` F16[4] at register 0 and `luminanceFactor` F16[4] at register 2,
    // and its own header says the buffer is 4 registers / 16 bytes. Counting a
    // component as a register reported 24, and the extra 8 bytes were read out of
    // the uniform RING - the previous draw's values, which drift smoothly frame
    // over frame and read exactly like a runaway multiplier the guest is computing.
    assert_eq!(uniform_bytes(&[(1, 4, 1, 0), (1, 4, 1, 2)]), 16);
    // An F16 pair is one register; an odd component count still rounds up to one.
    assert_eq!(uniform_bytes(&[(1, 2, 1, 0)]), 4);
    assert_eq!(uniform_bytes(&[(1, 1, 1, 0)]), 4);
    assert_eq!(uniform_bytes(&[(1, 3, 1, 5)]), 28);
}

#[test]
fn narrow_integer_parameters_use_their_own_widths() {
    // U16/S16/C10 pack two per register, U8/S8 four - the same table as
    // `ParamType::component_bytes`.
    assert_eq!(uniform_bytes(&[(5, 4, 1, 0)]), 8);
    assert_eq!(uniform_bytes(&[(6, 4, 1, 0)]), 8);
    assert_eq!(uniform_bytes(&[(2, 4, 1, 0)]), 8);
    assert_eq!(uniform_bytes(&[(7, 4, 1, 0)]), 4);
    assert_eq!(uniform_bytes(&[(8, 8, 1, 0)]), 8);
}

#[test]
fn an_unknown_type_is_sized_as_wide_as_possible() {
    // A type nibble this table does not know gets one register per component. That
    // can only ever OVER-size, and an over-size is recoverable where a truncated
    // buffer silently drops the tail of a matrix.
    assert_eq!(uniform_bytes(&[(9, 4, 1, 0)]), 16);
    assert_eq!(uniform_bytes(&[(0xf, 2, 2, 0)]), 16);
}

#[test]
fn the_extent_is_the_maximum_not_the_sum() {
    // Two parameters that OVERLAP (a title may alias a block) size the buffer by
    // the furthest reach, not by adding them up.
    assert_eq!(uniform_bytes(&[(0, 4, 1, 0), (0, 4, 1, 0)]), 16);
    // ...and order in the table does not matter.
    assert_eq!(uniform_bytes(&[(0, 4, 1, 8), (0, 4, 1, 0)]), 48);
}
