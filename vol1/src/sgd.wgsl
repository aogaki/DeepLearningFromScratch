@group(0) @binding(0) var<uniform> lr: f32;
@group(0) @binding(1) var<storage, read_write> param: array<f32>;
@group(0) @binding(2) var<storage, read> grad: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&param)) {
        param[i] = param[i] - lr * grad[i];
    }
}