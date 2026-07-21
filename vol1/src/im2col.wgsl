// im2col: 1 スレッド = col 行列の 1 要素。(row, col) から (n,oy,ox,c,fy,fx) を
// div/mod で復元し、対応する入力画素(pad はみ出しは 0)を書き込む
struct Params {
    n: u32, c: u32, h: u32, w: u32,
    oh: u32, ow: u32, fh: u32, fw: u32,
    stride: u32, pad: u32, _p0: u32, _p1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> col: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;   // (n, oy, ox) を平坦化した行
    let cidx = gid.x;  // (c, fy, fx) を平坦化した列
    let rows = p.n * p.oh * p.ow;
    let cols = p.c * p.fh * p.fw;
    if (row >= rows || cidx >= cols) {
        return;
    }
    let n = row / (p.oh * p.ow);
    let rem = row % (p.oh * p.ow);
    let oy = rem / p.ow;
    let ox = rem % p.ow;
    let c = cidx / (p.fh * p.fw);
    let frem = cidx % (p.fh * p.fw);
    let fy = frem / p.fw;
    let fx = frem % p.fw;

    // pad があるので符号付きで(u32 のままだと負のはみ出しがラップする)
    let iy = i32(oy * p.stride + fy) - i32(p.pad);
    let ix = i32(ox * p.stride + fx) - i32(p.pad);

    var v = 0.0;
    if (iy >= 0 && iy < i32(p.h) && ix >= 0 && ix < i32(p.w)) {
        v = x[((n * p.c + c) * p.h + u32(iy)) * p.w + u32(ix)];
    }
    col[row * cols + cidx] = v;
}
