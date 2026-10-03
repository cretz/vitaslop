fn gxp_q2(v: vec2<f32>) -> vec2<f32> {
  let c = select(v, clamp(v, vec2<f32>(-65504.0), vec2<f32>(65504.0)), abs(v) <= vec2<f32>(3.40282346638528859812e+38f));
  return unpack2x16float(pack2x16float(vec2<f32>(vec2<f16>(c))));
}
