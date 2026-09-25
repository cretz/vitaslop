fn gxp_f16b(v: f32) -> u32 {
  let f = bitcast<u32>(v);
  let sign = (f >> 16u) & 0x8000u;
  let mag = f & 0x7fffffffu;
  let e = mag >> 23u;
  let nb = ((max(e, 112u) - 112u) << 10u) | ((mag >> 13u) & 0x3ffu);
  let nrem = mag & 0x1fffu;
  let normal = nb + select(0u, 1u, nrem > 0x1000u || (nrem == 0x1000u && (nb & 1u) == 1u));
  let sh = select(14u, 126u - e, e < 113u && e >= 102u);
  let m = (mag & 0x7fffffu) | 0x800000u;
  let sb = m >> sh;
  let half = 1u << (sh - 1u);
  let srem = m & ((1u << sh) - 1u);
  let sub = sb + select(0u, 1u, srem > half || (srem == half && (sb & 1u) == 1u));
  var r = select(sub, normal, e >= 113u);
  r = select(r, 0u, mag < 0x33000000u);
  r = select(r, 0x7bffu, mag >= 0x477ff000u);
  r = select(r, 0x7c00u, mag == 0x7f800000u);
  r = select(r, 0x7e00u, mag > 0x7f800000u);
  return sign | r;
}
fn gxp_hpk(lo: f32, hi: f32) -> u32 { return gxp_f16b(lo) | (gxp_f16b(hi) << 16u); }
