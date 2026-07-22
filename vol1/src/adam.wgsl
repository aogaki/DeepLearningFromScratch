struct AdamUniforms {
    lr: f32,
    c1: f32,
    c2: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> uniforms: AdamUniforms;
@group(0) @binding(1) var<storage, read_write> param: array<f32>;
@group(0) @binding(2) var<storage, read> grad: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;

const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPSILON: f32 = 1e-7;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&param)) {
        let g = grad[i];
        let m_new = BETA1 * m[i] + (1.0 - BETA1) * g;
        let v_new = BETA2 * v[i] + (1.0 - BETA2) * g * g;

        m[i] = m_new;
        v[i] = v_new;

        let m_hat = m_new * uniforms.c1;
        let v_hat = v_new * uniforms.c2;

        param[i] = param[i] - uniforms.lr * m_hat / (sqrt(v_hat) + EPSILON);
    }
}
