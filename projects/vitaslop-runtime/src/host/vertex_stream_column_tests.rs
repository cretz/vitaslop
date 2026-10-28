use super::*;

/// **A STREAM IS STEPPED BY ITS OWN STRIDE, EVEN WHEN AN ATTRIBUTE OVERHANGS IT.** One title's
/// world programs declare four attributes, the last an `UNTYPED x2` (eight bytes) at offset 32,
/// over a stream of stride 36 - and the buffer really is laid out at 36. The column was
/// widened to the attribute extent (40), which is the stride a single-stream draw reads the
/// guest's buffer at, so every vertex after the first came from the wrong place. A stride-0
/// stream still takes its column from its attributes.
#[test]
fn an_overhanging_attribute_does_not_widen_a_real_stride() {
    use crate::capture::VertexAttribute;
    let attr = |offset: u16, format: u8, component_count: u8, reg_index: u16| VertexAttribute {
        stream_index: 0,
        offset,
        format,
        component_count,
        reg_index,
    };
    let attrs = vec![attr(0, 9, 4, 0), attr(16, 10, 2, 4), attr(24, 10, 2, 6), attr(32, 10, 2, 8)];
    let mut st = VitaState::new(0, 64, Box::new(crate::world::DeterministicWorld::default()));
    st.set_vertex_program(1, attrs.clone(), vec![(36, false)], 0);
    assert_eq!(
        st.shader_patcher.vertex_program(1).unwrap().packed_stride,
        36,
        "the guest's stride, not the attribute extent"
    );
    // A stride-0 stream (one row for every vertex) still gets the room its attributes need.
    st.set_vertex_program(2, attrs, vec![(0, false)], 0);
    assert_eq!(st.shader_patcher.vertex_program(2).unwrap().packed_stride, 40);
}
