// 一時プローブ(Claude): matmul_tn + column_sum + affine backward 一式の検証。確認後に削除する
use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use vol1::gpu::{Gpu, GpuTensor};
use vol1::layers::AffineLayer;
use wgpu::util::DeviceExt;

const MATMUL_TN: &str = r#"
struct Dims { m: u32, k: u32, n: u32, _pad: u32 }

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row0 = gid.y * 4u;
    let col0 = gid.x * 4u;
    if (row0 >= dims.m || col0 >= dims.n) {
        return;
    }
    let k = dims.k;
    let m = dims.m;
    let n = dims.n;

    if (row0 + 4u <= m && col0 + 4u <= n) {
        var acc0 = vec4<f32>();
        var acc1 = vec4<f32>();
        var acc2 = vec4<f32>();
        var acc3 = vec4<f32>();
        for (var i = 0u; i < k; i = i + 1u) {
            let bi = i * n + col0;
            let bv = vec4<f32>(b[bi], b[bi + 1u], b[bi + 2u], b[bi + 3u]);
            let ai = i * m + row0;
            let av = vec4<f32>(a[ai], a[ai + 1u], a[ai + 2u], a[ai + 3u]);
            acc0 = acc0 + bv * av.x;
            acc1 = acc1 + bv * av.y;
            acc2 = acc2 + bv * av.z;
            acc3 = acc3 + bv * av.w;
        }
        let o0 = (row0 + 0u) * n + col0;
        out[o0] = acc0.x; out[o0 + 1u] = acc0.y; out[o0 + 2u] = acc0.z; out[o0 + 3u] = acc0.w;
        let o1 = (row0 + 1u) * n + col0;
        out[o1] = acc1.x; out[o1 + 1u] = acc1.y; out[o1 + 2u] = acc1.z; out[o1 + 3u] = acc1.w;
        let o2 = (row0 + 2u) * n + col0;
        out[o2] = acc2.x; out[o2 + 1u] = acc2.y; out[o2 + 2u] = acc2.z; out[o2 + 3u] = acc2.w;
        let o3 = (row0 + 3u) * n + col0;
        out[o3] = acc3.x; out[o3 + 1u] = acc3.y; out[o3 + 2u] = acc3.z; out[o3 + 3u] = acc3.w;
    } else {
        for (var r = 0u; r < 4u; r = r + 1u) {
            let row = row0 + r;
            if (row >= m) { break; }
            for (var c = 0u; c < 4u; c = c + 1u) {
                let col = col0 + c;
                if (col >= n) { break; }
                var sum = 0.0;
                for (var i = 0u; i < k; i = i + 1u) {
                    sum = sum + a[i * m + row] * b[i * n + col];
                }
                out[row * n + col] = sum;
            }
        }
    }
}
"#;

const COLUMN_SUM: &str = r#"
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
"#;

fn make_pipeline(device: &wgpu::Device, src: &str) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn matmul_tn(gpu: &Gpu, p: &wgpu::ComputePipeline, a: &GpuTensor, b: &GpuTensor) -> GpuTensor {
    let (k, m) = a.shape;
    let (k2, n) = b.shape;
    assert_eq!(k, k2);
    let dims: [u32; 4] = [m as u32, k as u32, n as u32, 0];
    let dims_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let out = GpuTensor {
        buffer: gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (m * n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }),
        shape: (m, n),
    };
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &p.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: a.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: b.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out.buffer.as_entire_binding() },
        ],
    });
    let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(p);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n.div_ceil(32) as u32, m.div_ceil(32) as u32, 1);
    }
    gpu.queue.submit(Some(enc.finish()));
    out
}

fn column_sum(gpu: &Gpu, p: &wgpu::ComputePipeline, x: &GpuTensor) -> GpuTensor {
    let (rows, cols) = x.shape;
    let dims: [u32; 4] = [rows as u32, cols as u32, 0, 0];
    let dims_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let out = GpuTensor {
        buffer: gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (cols * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }),
        shape: (1, cols),
    };
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &p.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: dims_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out.buffer.as_entire_binding() },
        ],
    });
    let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(p);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((cols as u32).div_ceil(64), 1, 1);
    }
    gpu.queue.submit(Some(enc.finish()));
    out
}

fn max_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn main() {
    let gpu = Gpu::new();
    let tn = make_pipeline(&gpu.device, MATMUL_TN);
    let cs = make_pipeline(&gpu.device, COLUMN_SUM);

    // 1. matmul_tn: C = Aᵀ·B、素数サイズ + affine 実寸
    for &(k, m, n) in &[(37usize, 53usize, 29usize), (5, 4, 4), (1000, 100, 50), (17, 3, 2)] {
        let a: Array2<f32> = Array2::random((k, m), StandardNormal);
        let b: Array2<f32> = Array2::random((k, n), StandardNormal);
        let expected = a.t().dot(&b);
        let got = gpu.download(&matmul_tn(&gpu, &tn, &gpu.upload(&a), &gpu.upload(&b)));
        let d = max_diff(&got, &expected);
        assert!(got.dim() == expected.dim() && d < 1e-3, "tn ({k},{m},{n}): {d:e}");
        println!("matmul_tn OK ({k},{m},{n}): max_diff {d:e}");
    }

    // 2. column_sum
    let x: Array2<f32> = Array2::random((37, 29), StandardNormal);
    let expected = x.sum_axis(Axis(0)).insert_axis(Axis(0)).into_owned();
    let got = gpu.download(&column_sum(&gpu, &cs, &gpu.upload(&x)));
    let d = max_diff(&got, &expected);
    assert!(d < 1e-4, "column_sum: {d:e}");
    println!("column_sum OK: max_diff {d:e}");

    // 3. affine backward 一式 vs CPU AffineLayer
    let xa: Array2<f32> = Array2::random((37, 53), StandardNormal);
    let wa: Array2<f32> = Array2::random((53, 29), StandardNormal);
    let ba: Array2<f32> = Array2::random((1, 29), StandardNormal);
    let dout: Array2<f32> = Array2::random((37, 29), StandardNormal);

    let mut cpu = AffineLayer::new(wa.clone(), ba.clone());
    let _ = cpu.forward(xa.clone());
    let cpu_dx = cpu.backward(dout.clone());

    // GPU: dW = xᵀ·dout (TN) / dx = dout·wᵀ (通常 matmul、wᵀ は事前 upload) / db = 列和
    let g_x = gpu.upload(&xa);
    let g_dout = gpu.upload(&dout);
    let wt: Array2<f32> = wa.t().as_standard_layout().into_owned();
    let g_wt = gpu.upload(&wt);

    let gpu_dw = gpu.download(&matmul_tn(&gpu, &tn, &g_x, &g_dout));
    let gpu_dx = gpu.download(&gpu.matmul_gpu(&g_dout, &g_wt));
    let gpu_db = gpu.download(&column_sum(&gpu, &cs, &g_dout));

    let d_dw = max_diff(&gpu_dw, cpu.dw());
    let d_dx = max_diff(&gpu_dx, &cpu_dx);
    let d_db = max_diff(&gpu_db, cpu.db());
    assert!(d_dw < 1e-3 && d_dx < 1e-3 && d_db < 1e-3, "dw {d_dw:e} dx {d_dx:e} db {d_db:e}");
    println!("affine backward OK: dw {d_dw:e} / dx {d_dx:e} / db {d_db:e}");
}
