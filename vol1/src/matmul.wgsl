// matmul (vec4 レジスタタイリング): 1 スレッドが 4x4 出力を担当。
// 蓄積は vec4 レジスタ(ローカル配列は使わない — 動的添字はスレッド私有メモリに落ちる)。
// バリアが無いので early return が合法(barrier 有りのカーネルでは不可)
struct Dims { m: u32, k: u32, n: u32, _pad: u32 }

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row0 = gid.y * 4u;
    let col0 = gid.x * 4u;
    if (row0 >= dims.m || col0 >= dims.n) {
        return;
    }
    let k = dims.k;
    let n = dims.n;

    if (row0 + 4u <= dims.m && col0 + 4u <= dims.n) {
        // 内側フルブロック: 4x4 を vec4 レジスタ 4 本で蓄積
        var acc0 = vec4<f32>();
        var acc1 = vec4<f32>();
        var acc2 = vec4<f32>();
        var acc3 = vec4<f32>();
        for (var i = 0u; i < k; i = i + 1u) {
            let bi = i * n + col0;
            let bv = vec4<f32>(b[bi], b[bi + 1u], b[bi + 2u], b[bi + 3u]);
            acc0 = acc0 + bv * a[(row0 + 0u) * k + i];
            acc1 = acc1 + bv * a[(row0 + 1u) * k + i];
            acc2 = acc2 + bv * a[(row0 + 2u) * k + i];
            acc3 = acc3 + bv * a[(row0 + 3u) * k + i];
        }
        let o0 = (row0 + 0u) * n + col0;
        out[o0] = acc0.x; out[o0 + 1u] = acc0.y; out[o0 + 2u] = acc0.z; out[o0 + 3u] = acc0.w;
        let o1 = (row0 + 1u) * n + col0;
        out[o1] = acc1.x; out[o1 + 1u] = acc1.y; out[o1 + 2u] = acc1.z; out[o1 + 3u] = acc1.w;
        let o2 = (row0 + 2u) * n + col0;
        out[o2] = acc2.x; out[o2 + 1u] = acc2.y; out[o2 + 2u] = acc2.z; out[o2 + 3u] = acc2.w;
        let o3 = (row0 + 3u) * n + col0;
        out[o3] = acc3.x; out[o3 + 1u] = acc3.y; out[o3 + 2u] = acc3.z; out[o3 + 3u] = acc3.w;
    } else {
        // 端の欠けブロック: スカラーで処理(端の workgroup だけが通る)
        for (var r = 0u; r < 4u; r = r + 1u) {
            let row = row0 + r;
            if (row >= dims.m) { break; }
            for (var c = 0u; c < 4u; c = c + 1u) {
                let col = col0 + c;
                if (col >= dims.n) { break; }
                var sum = 0.0;
                for (var i = 0u; i < k; i = i + 1u) {
                    sum = sum + a[row * k + i] * b[i * n + col];
                }
                out[row * n + col] = sum;
            }
        }
    }
}
