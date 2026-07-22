// forward で使った nhwc_to_nchw の逆写像(backward の dout を行列の形に戻す)

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
    // i を NHWC 側 (n, oy, ox, f) と解釈して、NCHW 側から拾う
    let f = i % p.f;
    let r1 = i / p.f;
    let ox = r1 % p.ow;
    let r2 = r1 / p.ow;
    let oy = r2 % p.oh;
    let n = r2 / p.oh;
    dst[i] = src[((n * p.f + f) * p.oh + oy) * p.ow + ox];
}