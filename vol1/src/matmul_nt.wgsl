struct Dims {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x; // n 次元
    let row = gid.y; // m 次元

    if (row >= dims.m || col >= dims.n) {
        return;
    }

    let k = dims.k;
    let a_offset = row * k;
    let b_offset = col * k;

    var acc = vec4<f32>(0.0);
    let k4 = k / 4u;

    // k軸に沿って両行列から4要素ずつ連続読み出し
    for (var i = 0u; i < k4; i = i + 1u) {
        let a_idx = a_offset + i * 4u;
        let b_idx = b_offset + i * 4u;

        let av = vec4<f32>(a[a_idx], a[a_idx+1u], a[a_idx+2u], a[a_idx+3u]);
        let bv = vec4<f32>(b[b_idx], b[b_idx+1u], b[b_idx+2u], b[b_idx+3u]);
        
        acc += av * bv;
    }

    // 水平和
    var sum = acc.x + acc.y + acc.z + acc.w;

    // 端数処理
    for (var i = k4 * 4u; i < k; i = i + 1u) {
        sum += a[a_offset + i] * b[b_offset + i];
    }

    c[row * dims.n + col] = sum;
}