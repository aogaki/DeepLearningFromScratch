struct Dims {
    n: u32, c: u32, h: u32, w: u32,
    oh: u32, ow: u32, ph: u32, pw: u32,
    stride: u32, pad: u32, _pad1: u32, _pad2: u32
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> dout: array<f32>;
@group(0) @binding(2) var<storage, read> argmax: array<u32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total_out = dims.n * dims.c * dims.oh * dims.ow;
    if (i >= total_out) {
        return;
    }

    // 出力位置(1D)から多次元インデックスの復元
    let ow = dims.ow;
    let oh = dims.oh;
    let c = dims.c;
    
    let ox = i % ow;
    let oy = (i / ow) % oh;
    let c_ = (i / (ow * oh)) % c;
    let n_ = i / (c * ow * oh);

    // argmax からウィンドウ内ローカル座標 (fy, fx) を復元
    let a = argmax[i];
    let fy = a / dims.pw;
    let fx = a % dims.pw;

    // 入力画像上のグローバル座標 (iy, ix) を計算
    // pad がある場合、有効な argmax は必ず元画像内を指すため
    // 安全のため i32 で計算してから u32 に戻して範囲チェック
    let iy = i32(oy * dims.stride + fy) - i32(dims.pad);
    let ix = i32(ox * dims.stride + fx) - i32(dims.pad);

    if (iy >= 0 && iy < i32(dims.h) && ix >= 0 && ix < i32(dims.w)) {
        let dx_idx = ((n_ * c + c_) * dims.h + u32(iy)) * dims.w + u32(ix);
        dx[dx_idx] = dout[i];
    }
}