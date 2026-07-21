// pad 領域は -inf 扱い(= max から除外)。本の CPU 版は im2col 流用のため 0 埋めで、
// 「窓内が全負のとき 0 が勝つ」挙動をするが、こちらが標準的な max-pool の定義
// (DeepConvNet は pool pad=0 なので実運用では差は出ない)
struct Params {
    n: u32, c: u32, h: u32, w: u32,
    oh: u32, ow: u32, ph: u32, pw: u32,
    stride: u32, pad: u32, _p0: u32, _p1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read_write> argmax: array<u32>; // backward用の記録

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.n * p.c * p.oh * p.ow;
    if (idx >= total) {
        return;
    }

    // idx は (N, C, OH, OW) の NCHW 平坦化レイアウトのインデックス
    let ox = idx % p.ow;
    let oy = (idx / p.ow) % p.oh;
    let c  = (idx / (p.oh * p.ow)) % p.c;
    let n  = idx / (p.c * p.oh * p.ow);

    // f32 の最小値 (-3.40282347E+38) で初期化
    var max_val = -3.402823e+38; 
    var max_idx = 0u;
    
    for (var fy = 0u; fy < p.ph; fy++) {
        for (var fx = 0u; fx < p.pw; fx++) {
            let iy = i32(oy * p.stride + fy) - i32(p.pad);
            let ix = i32(ox * p.stride + fx) - i32(p.pad);

            var v = -3.402823e+38;
            if (iy >= 0 && iy < i32(p.h) && ix >= 0 && ix < i32(p.w)) {
                let in_idx = ((n * p.c + c) * p.h + u32(iy)) * p.w + u32(ix);
                v = x[in_idx];
            }

            // 最大値と、そのパッチ内でのローカルインデックスを記録
            if (v > max_val) {
                max_val = v;
                max_idx = fy * p.pw + fx;
            }
        }
    }

    out[idx] = max_val;
    argmax[idx] = max_idx;
}