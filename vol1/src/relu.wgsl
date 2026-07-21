// relu.wgsl — 要素ごと in-place ReLU
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&data)) {
        data[i] = max(data[i], 0.0);
    }
}