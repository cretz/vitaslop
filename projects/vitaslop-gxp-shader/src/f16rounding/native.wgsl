fn gxp_f16c(v: f32) -> f32 {
  let m = bitcast<u32>(v) & 0x7fffffffu;
  if (m > 0x477fe000u && m < 0x7f800000u) { return sign(v) * 65504.0; }
  return v;
}
fn gxp_f16r(v: f32) -> f32 { return f32(f16(gxp_f16c(v))); }
fn gxp_f16b(v: f32) -> u32 { return pack2x16float(vec2<f32>(gxp_f16r(v), 0.0)) & 0x0000ffffu; }
fn gxp_hpk(lo: f32, hi: f32) -> u32 {
  return pack2x16float(vec2<f32>(gxp_f16r(lo), gxp_f16r(hi)));
}
