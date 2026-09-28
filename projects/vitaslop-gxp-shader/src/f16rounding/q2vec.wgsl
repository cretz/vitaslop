fn gxp_q2(v: vec2<f32>) -> vec2<f32> {
  let m = bitcast<vec2<u32>>(v) & vec2<u32>(0x7fffffffu);
  let c = select(v, sign(v) * 65504.0, (m > vec2<u32>(0x477fe000u)) & (m < vec2<u32>(0x7f800000u)));
  return unpack2x16float(pack2x16float(vec2<f32>(vec2<f16>(c))));
}
