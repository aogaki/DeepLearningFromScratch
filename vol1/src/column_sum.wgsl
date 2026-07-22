// 1 スレッド = 1 列なので、db 計算(列数 10〜64)では数十スレッドしか走らない超低並列です。
// ただし仕事量自体が誤差(全体の 0.1% 未満)なので今は これで良し。
// 将来効いてきたら「二段リダクション」という古典で直す
struct Dims { rows: u32, cols: u32, _p0: u32, _p1: u32 }

@group(0) @binding(0) var<uniform> d: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= d.cols) {
        return;
    }
    var s = 0.0;
    for (var r = 0u; r < d.rows; r = r + 1u) {
        s = s + x[r * d.cols + c];
    }
    out[c] = s;
}