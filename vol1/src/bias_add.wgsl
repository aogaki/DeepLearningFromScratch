// bias_add.wgsl — x(m,n) の各行に bias(1,n) を加算(in-place)。
// n は uniform で渡さず bias の arrayLength から得る(i % n で列位置に写す)
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&x)) {
        x[i] = x[i] + bias[i % arrayLength(&bias)];
    }
}