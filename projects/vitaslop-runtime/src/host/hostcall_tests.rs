//! Verify the `#[hostcall]` marshalling directly (the cube exercises only
//! integer/pointer handlers, so these cover the hardfloat classification and
//! the float returns end to end against a hand-built `GuestCtx`).
use super::*;
use crate::hostcall;
use crate::world::DeterministicWorld;

fn dummy_state() -> VitaState {
    VitaState::new(0, 64, Box::new(DeterministicWorld::default()))
}

// Mixed integer and float args with a float return: `a` and `b` come from the
// core registers (r0, r1), `x` and `y` from the VFP file (s0 and d1), and the
// f32 result goes to s0.
#[hostcall]
fn mix(st: &mut VitaState, a: u32, x: f32, b: i32, y: f64) -> f32 {
    a as f32 + x + b as f32 + y as f32 + st.base as f32
}

// A pure float handler: double the d0 argument back into d0.
#[hostcall]
fn dbl(v: f64) -> f64 {
    v * 2.0
}

// A pointer out-param handler: write the sum of two ints through the pointer.
#[hostcall]
fn sum_out(ctx: &mut GuestCtx, a: u32, b: u32, out: Ptr) -> i32 {
    ctx.write_u32(out.addr(), a.wrapping_add(b));
    0
}

#[test]
fn classifies_int_and_float_args_and_returns_float() {
    let mut regs = [0u32; REG_COUNT];
    let mut vfp = [0u32; VFP_ARG_COUNT];
    let mut bytes = vec![0u8; 64];
    regs[0] = 5; // a
    regs[1] = (-3i32) as u32; // b
    vfp[0] = 2.5f32.to_bits(); // x -> s0
    let y = 1.5f64.to_bits(); // y -> d1 (s2 low, s3 high)
    vfp[2] = y as u32;
    vfp[3] = (y >> 32) as u32;

    let mut st = dummy_state();
    let mut mem = SliceMemory(&mut bytes);
    {
        let mut ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
        mix(&mut ctx, &mut st);
    }
    // 5 + 2.5 + (-3) + 1.5 + 0 = 6.0, in s0.
    assert_eq!(f32::from_bits(vfp[0]), 6.0);
}

#[test]
fn marshals_double_arg_and_return() {
    let mut regs = [0u32; REG_COUNT];
    let mut vfp = [0u32; VFP_ARG_COUNT];
    let mut bytes = vec![0u8; 8];
    let v = 3.0f64.to_bits(); // d0 = s0 (low), s1 (high)
    vfp[0] = v as u32;
    vfp[1] = (v >> 32) as u32;

    let mut st = dummy_state();
    let mut mem = SliceMemory(&mut bytes);
    {
        let mut ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
        dbl(&mut ctx, &mut st);
    }
    let out = (vfp[0] as u64) | ((vfp[1] as u64) << 32);
    assert_eq!(f64::from_bits(out), 6.0);
}

#[test]
fn writes_through_pointer_out_param() {
    let mut regs = [0u32; REG_COUNT];
    let mut vfp = [0u32; VFP_ARG_COUNT];
    let mut bytes = vec![0u8; 64];
    regs[0] = 7; // a
    regs[1] = 35; // b
    regs[2] = 16; // out pointer (guest addr, base 0 -> offset 16)

    let mut st = dummy_state();
    let mut mem = SliceMemory(&mut bytes);
    {
        let mut ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
        sum_out(&mut ctx, &mut st);
    }
    assert_eq!(
        u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
        42
    );
}
