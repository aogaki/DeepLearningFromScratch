// matmul 出力 (N·OH·OW, F) を NCHW 平坦 (N, F·OH·OW) へ並べ替える。
// 1 スレッド = 出力 1 要素: 自分の NCHW 座標 (n,f,oy,ox) を復元し、NHWC 側から拾う
struct P { n: u32, f: u32, oh: u32, ow: u32 }

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = p.n * p.f * p.oh * p.ow;
    if (i >= total) {
        return;
    }
    let n = i / (p.f * p.oh * p.ow);
    let r1 = i % (p.f * p.oh * p.ow);
    let f = r1 / (p.oh * p.ow);
    let r2 = r1 % (p.oh * p.ow);
    let oy = r2 / p.ow;
    let ox = r2 % p.ow;
    dst[i] = src[((n * p.oh + oy) * p.ow + ox) * p.f + f];
}
