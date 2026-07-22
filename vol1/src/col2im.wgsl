// 入力画素ごとの gather: 自分を参照した col 要素を逆算して集める(atomic 不要・決定論)

struct Params {
    n: u32, c: u32, h: u32, w: u32,
    oh: u32, ow: u32, fh: u32, fw: u32,
    stride: u32, pad: u32, _p0: u32, _p1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> dcol: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = p.n * p.c * p.h * p.w;
    if (i >= total) {
        return;
    }
    // i を NCHW (n, c, iy, ix) と解釈
    let ix = i % p.w;
    let iy = (i / p.w) % p.h;
    let c = (i / (p.h * p.w)) % p.c;
    let n = i / (p.c * p.h * p.w);

    // iy = oy*stride + fy - pad かつ 0 <= fy < fh を満たす oy の区間を求める。
    // t = iy + pad >= oy*stride かつ oy*stride > t - fh
    // (u32 のまま扱うため、負になり得る下限は「t+1 <= fh なら 0」で場合分け)
    let ty = iy + p.pad;
    let tx = ix + p.pad;
    var oy_min = 0u;
    if (ty + 1u > p.fh) {
        oy_min = (ty + 1u - p.fh + p.stride - 1u) / p.stride; // ceil 割り
    }
    var ox_min = 0u;
    if (tx + 1u > p.fw) {
        ox_min = (tx + 1u - p.fw + p.stride - 1u) / p.stride;
    }
    let oy_max = min(ty / p.stride, p.oh - 1u);
    let ox_max = min(tx / p.stride, p.ow - 1u);

    let cols = p.c * p.fh * p.fw;
    var s = 0.0;
    // oy_min > oy_max のとき(この画素を覆う窓が無い)はループが回らず s = 0
    for (var oy = oy_min; oy <= oy_max; oy = oy + 1u) {
        let fy = ty - oy * p.stride;
        for (var ox = ox_min; ox <= ox_max; ox = ox + 1u) {
            let fx = tx - ox * p.stride;
            let row = (n * p.oh + oy) * p.ow + ox;
            let col_idx = (c * p.fh + fy) * p.fw + fx;
            s = s + dcol[row * cols + col_idx];
        }
    }
    dx[i] = s;
}