@group(0) @binding(0) var<storage, read_write> dout: array<f32>;
@group(0) @binding(1) var<storage, read> act: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&dout)) {
        dout[i] = dout[i] * select(0.0, 1.0, act[i] > 0.0);
    }
}
