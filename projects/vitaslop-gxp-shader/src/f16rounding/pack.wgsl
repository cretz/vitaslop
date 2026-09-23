fn gxp_f16b(v: f32) -> u32 { return pack2x16float(vec2<f32>(v, 0.0)) & 0x0000ffffu; }
fn gxp_hpk(lo: f32, hi: f32) -> u32 { return pack2x16float(vec2<f32>(lo, hi)); }
