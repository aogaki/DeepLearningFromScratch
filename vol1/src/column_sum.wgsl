struct Uniforms {
    m: u32,
    n: u32,
}
// Rust 側の配置に合わせた正しい順番
@group(0) @binding(0) var<uniform> dims: Uniforms;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> b: array<f32>;

// 共有の黒板
var<workgroup> partial: array<f32, 256>;

// 1つの出力列(col) に対して、256人のスレッドチームを起動
@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,       // 担当する列番号 (col)
    @builtin(local_invocation_id) local_id: vec3<u32> // スレッド番号 0〜255 (lid)
) {
    let col = group_id.x;
    let lid = local_id.x;

    if (col >= dims.n) { return; }

    // ====== 第1幕: stride 分担の私的部分和 ======
    var sum = 0.0;
    // 対象列の m 行を 256 人で均等に手分けして縦に足し下ろす
    for (var i = lid; i < dims.m; i = i + 256u) {
        sum = sum + a[i * dims.n + col];
    }

    // ====== 第2幕: 共有の黒板に書き出す ======
    partial[lid] = sum;
    workgroupBarrier(); // 全員が書き終わるのを待つ

    // ====== 第3幕: 木構造の畳み込み ======
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            partial[lid] = partial[lid] + partial[lid + s];
        }
        workgroupBarrier(); // ラウンドごとの読み書きを同期
    }

    // 最後にスレッド0が、1列ぶんの最終合計を VRAM に書き戻す
    if (lid == 0u) {
        b[col] = partial[0];
    }
}