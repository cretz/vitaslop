fn gxp_f16c(v: f32) -> f32 {
  return select(v, clamp(v, -65504.0, 65504.0), abs(v) <= 3.40282346638528859812e+38f);
}
fn gxp_f16r(v: f32) -> f32 { return f32(f16(gxp_f16c(v))); }
fn gxp_f16b(v: f32) -> u32 { return pack2x16float(vec2<f32>(gxp_f16r(v), 0.0)) & 0x0000ffffu; }
fn gxp_hpk(lo: f32, hi: f32) -> u32 {
  return pack2x16float(vec2<f32>(gxp_f16r(lo), gxp_f16r(hi)));
}
