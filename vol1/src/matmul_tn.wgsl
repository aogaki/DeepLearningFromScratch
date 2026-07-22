struct Uniforms {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

// Rust 側の配置に合わせた正しい順番
@group(0) @binding(0) var<uniform> dims: Uniforms;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

// 共有の黒板
var<workgroup> partial: array<f32, 256>;

// 1つの出力マス(row, col) に対して、256人のスレッドチームを起動
@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,       // 担当する出力マス (row, col)
    @builtin(local_invocation_id) local_id: vec3<u32> // スレッド番号 0〜255 (lid)
) {
    let row = group_id.x;
    let col = group_id.y;
    let lid = local_id.x; // z上限(64)を避けるため x 次元を使用

    if (row >= dims.m || col >= dims.n) { return; }

    // ====== 第1幕: stride 分担の私的部分和 ======
    var sum = 0.0;
    // k 行にわたって足し上げる (例: 78,400)
    for (var i = lid; i < dims.k; i = i + 256u) {
        sum = sum + a[i * dims.m + row] * b[i * dims.n + col];
    }

    // ====== 第2幕: 共有の黒板に書き出す ======
    partial[lid] = sum;
    workgroupBarrier();

    // ====== 第3幕: 木構造の畳み込み ======
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            partial[lid] = partial[lid] + partial[lid + s];
        }
        workgroupBarrier();
    }

    // 最後にスレッド0が、1マスぶんの最終結果を VRAM に書き戻す
    if (lid == 0u) {
        c[row * dims.n + col] = partial[0];
    }
}
