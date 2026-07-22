struct Uniforms {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Uniforms;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

// 共有の黒板: 4本 × 256 × 16バイト(vec4) = 16,384 バイト
// 注意: これは WebGPU のデフォルト上限 maxComputeWorkgroupStorageSize ぴったりであり、ヘッドルームは 0 バイト。
// 将来的に容量が必要になった場合の脱出路は「配列を2本に減らし、木の畳み込みを2周回す」こと。
var<workgroup> p0: array<vec4<f32>, 256>;
var<workgroup> p1: array<vec4<f32>, 256>;
var<workgroup> p2: array<vec4<f32>, 256>;
var<workgroup> p3: array<vec4<f32>, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row0 = group_id.x * 4u;
    let col0 = group_id.y * 4u;
    let lid = local_id.x;

    let m = dims.m;
    let n = dims.n;

    // 端数ブロックでも配列外アクセスを起こさないよう添字をクランプ (読んだゴミは最終書き出しで捨てる)
    let r0 = row0;
    let r1 = min(row0 + 1u, m - 1u);
    let r2 = min(row0 + 2u, m - 1u);
    let r3 = min(row0 + 3u, m - 1u);

    let c0 = col0;
    let c1 = min(col0 + 1u, n - 1u);
    let c2 = min(col0 + 2u, n - 1u);
    let c3 = min(col0 + 3u, n - 1u);

    var acc0 = vec4<f32>();
    var acc1 = vec4<f32>();
    var acc2 = vec4<f32>();
    var acc3 = vec4<f32>();

    // ====== 第1幕: stride 分担の私的部分和 (1ループで16積和・8フェッチ) ======
    for (var i = lid; i < dims.k; i = i + 256u) {
        let ai = i * m;
        let av = vec4<f32>(a[ai + r0], a[ai + r1], a[ai + r2], a[ai + r3]);

        let bi = i * n;
        let bv = vec4<f32>(b[bi + c0], b[bi + c1], b[bi + c2], b[bi + c3]);
        
        acc0 = acc0 + bv * av.x;
        acc1 = acc1 + bv * av.y;
        acc2 = acc2 + bv * av.z;
        acc3 = acc3 + bv * av.w;
    }

    // ====== 第2幕: 共有の黒板に書き出す (4本分の vec4) ======
    p0[lid] = acc0;
    p1[lid] = acc1;
    p2[lid] = acc2;
    p3[lid] = acc3;
    workgroupBarrier();

    // ====== 第3幕: 木構造の畳み込み ======
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            p0[lid] = p0[lid] + p0[lid + s];
            p1[lid] = p1[lid] + p1[lid + s];
            p2[lid] = p2[lid] + p2[lid + s];
            p3[lid] = p3[lid] + p3[lid + s];
        }
        workgroupBarrier();
    }

    // ====== 最後にスレッド0が、4x4マスぶんの最終結果を VRAM に書き戻す (端数マスは弾く) ======
    if (lid == 0u) {
        if (row0 < m) {
            if (col0 < n) { c[row0 * n + col0] = p0[0].x; }
            if (col0 + 1u < n) { c[row0 * n + col0 + 1u] = p0[0].y; }
            if (col0 + 2u < n) { c[row0 * n + col0 + 2u] = p0[0].z; }
            if (col0 + 3u < n) { c[row0 * n + col0 + 3u] = p0[0].w; }
        }
        if (row0 + 1u < m) {
            let r_idx = row0 + 1u;
            if (col0 < n) { c[r_idx * n + col0] = p1[0].x; }
            if (col0 + 1u < n) { c[r_idx * n + col0 + 1u] = p1[0].y; }
            if (col0 + 2u < n) { c[r_idx * n + col0 + 2u] = p1[0].z; }
            if (col0 + 3u < n) { c[r_idx * n + col0 + 3u] = p1[0].w; }
        }
        if (row0 + 2u < m) {
            let r_idx = row0 + 2u;
            if (col0 < n) { c[r_idx * n + col0] = p2[0].x; }
            if (col0 + 1u < n) { c[r_idx * n + col0 + 1u] = p2[0].y; }
            if (col0 + 2u < n) { c[r_idx * n + col0 + 2u] = p2[0].z; }
            if (col0 + 3u < n) { c[r_idx * n + col0 + 3u] = p2[0].w; }
        }
        if (row0 + 3u < m) {
            let r_idx = row0 + 3u;
            if (col0 < n) { c[r_idx * n + col0] = p3[0].x; }
            if (col0 + 1u < n) { c[r_idx * n + col0 + 1u] = p3[0].y; }
            if (col0 + 2u < n) { c[r_idx * n + col0 + 2u] = p3[0].z; }
            if (col0 + 3u < n) { c[r_idx * n + col0 + 3u] = p3[0].w; }
        }
    }
}
