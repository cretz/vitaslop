fn gxp_f16b(v: f32) -> u32 {
  let f = bitcast<u32>(v);
  let sign = (f >> 16u) & 0x8000u;
  let mag = f & 0x7fffffffu;
  if (mag > 0x7f800000u) { return sign | 0x7e00u; }
  if (mag == 0x7f800000u) { return sign | 0x7c00u; }
  if (mag >= 0x477ff000u) { return sign | 0x7bffu; }
  if (mag < 0x33000000u) { return sign; }
  let e = mag >> 23u;
  if (e >= 113u) {
    let bits = ((e - 112u) << 10u) | ((mag >> 13u) & 0x3ffu);
    let rem = mag & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (bits & 1u) == 1u)) { return sign | (bits + 1u); }
    return sign | bits;
  }
  let shift = 126u - e;
  let m = (mag & 0x7fffffu) | 0x800000u;
  let bits = m >> shift;
  let half = 1u << (shift - 1u);
  let rem = m & ((1u << shift) - 1u);
  if (rem > half || (rem == half && (bits & 1u) == 1u)) { return sign | (bits + 1u); }
  return sign | bits;
}
fn gxp_hpk(lo: f32, hi: f32) -> u32 { return gxp_f16b(lo) | (gxp_f16b(hi) << 16u); }
