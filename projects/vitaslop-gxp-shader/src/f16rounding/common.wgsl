fn gxp_hlo(cur: u32, v: f32) -> u32 { return (cur & 0xffff0000u) | gxp_f16b(v); }
fn gxp_hhi(cur: u32, v: f32) -> u32 { return (cur & 0x0000ffffu) | (gxp_f16b(v) << 16u); }
fn gxp_hq(v: f32) -> f32 { return unpack2x16float(gxp_f16b(v))[0]; }
