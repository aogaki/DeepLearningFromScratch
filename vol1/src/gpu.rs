use ndarray::Array2;
use wgpu::{Device, Queue, util::DeviceExt};

const DOUBLE_SHADER: &str = r#"
@group(0) @binding(0)
var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&data)) {
        data[i] = data[i] * 2.0;
    }
}
"#;

pub struct GpuTensor {
    pub buffer: wgpu::Buffer,
    pub shape: (usize, usize),
}

pub struct Gpu {
    pub device: Device,
    pub queue: Queue,
    matmul_pipeline: wgpu::ComputePipeline,
    relu_pipeline: wgpu::ComputePipeline,
    bias_add_pipeline: wgpu::ComputePipeline,
}
impl Gpu {
    pub fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("No GPU adapter found");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("Failed to open GPU device");

        let matmul_pipeline = Self::make_pipeline(&device, include_str!("matmul.wgsl"), "matmul");
        let relu_pipeline = Self::make_pipeline(&device, include_str!("relu.wgsl"), "relu");
        let bias_add_pipeline =
            Self::make_pipeline(&device, include_str!("bias_add.wgsl"), "bias_add");

        Gpu {
            device,
            queue,
            matmul_pipeline,
            relu_pipeline,
            bias_add_pipeline,
        }
    }

    fn make_pipeline(device: &wgpu::Device, src: &str, label: &str) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    pub fn read_buffer(&self, buffer: &wgpu::Buffer) -> Vec<f32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: buffer.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("Poll failed");
        rx.recv().expect("Channel closed").expect("Map failed");

        let view = slice.get_mapped_range().expect("get_mapped_range failed");
        let out: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        staging.unmap();
        out
    }

    pub fn double(&self, data: &[f32]) -> Vec<f32> {
        let storage = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("double storage"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("double"),
                source: wgpu::ShaderSource::Wgsl(DOUBLE_SHADER.into()),
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("double"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: storage.as_entire_binding(),
            }],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(data.len().div_ceil(64) as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        self.read_buffer(&storage)
    }

    pub fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        self.download(&self.matmul_gpu(&self.upload(a), &self.upload(b)))
    }

    /// Array2 を GPU バッファへ転送して常駐テンソルにする
    pub fn upload(&self, a: &Array2<f32>) -> GpuTensor {
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tensor"),
                contents: bytemuck::cast_slice(a.as_slice().expect("standard layout")),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        GpuTensor {
            buffer,
            shape: a.dim(),
        }
    }

    /// 常駐テンソルを CPU に読み戻す
    pub fn download(&self, t: &GpuTensor) -> Array2<f32> {
        Array2::from_shape_vec(t.shape, self.read_buffer(&t.buffer)).expect("shape mismatch")
    }

    /// GPU 常駐 matmul: 入力も出力も GPU に置いたまま
    pub fn matmul_gpu(&self, a: &GpuTensor, b: &GpuTensor) -> GpuTensor {
        let (m, k) = a.shape;
        let (k2, n) = b.shape;
        assert_eq!(k, k2, "matmul_gpu: inner dimensions must match");

        let dims: [u32; 4] = [m as u32, k as u32, n as u32, 0];
        let dims_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dims"),
                contents: bytemuck::cast_slice(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: (m * n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.matmul_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dims_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.matmul_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n.div_ceil(32) as u32, m.div_ceil(32) as u32, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        GpuTensor {
            buffer: out_buf,
            shape: (m, n),
        }
    }

    /// 1 次元 dispatch の定型(bind group 構築 → pass → submit)
    fn dispatch_1d(
        &self,
        pipeline: &wgpu::ComputePipeline,
        entries: &[wgpu::BindGroupEntry],
        n_threads: usize,
    ) {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_threads.div_ceil(64) as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// GPU 上で in-place ReLU
    pub fn relu_gpu(&self, x: &mut GpuTensor) {
        let n = x.shape.0 * x.shape.1;
        self.dispatch_1d(
            &self.relu_pipeline,
            &[wgpu::BindGroupEntry {
                binding: 0,
                resource: x.buffer.as_entire_binding(),
            }],
            n,
        );
    }

    /// GPU 上で bias(1,n) を各行に加算(in-place)
    pub fn add_bias_gpu(&self, x: &mut GpuTensor, bias: &GpuTensor) {
        assert_eq!(bias.shape, (1, x.shape.1), "bias must be (1, n)");
        let n = x.shape.0 * x.shape.1;
        self.dispatch_1d(
            &self.bias_add_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bias.buffer.as_entire_binding(),
                },
            ],
            n,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::StandardNormal;
    use std::hint::black_box;
    use std::time::Instant;

    #[test]
    fn test_gpu_new() {
        let gpu = Gpu::new();
        // Ensure the device and queue are created
        assert!(gpu.device.limits().max_buffer_size > 0);
    }

    #[test]
    fn test_buffer_round_trip() {
        let gpu = Gpu::new();
        let input: Vec<f32> = (0..1024).map(|i| i as f32).collect();

        let storage = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("storage"),
                contents: bytemuck::cast_slice(&input),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let output = gpu.read_buffer(&storage);
        assert_eq!(input, output); // 移動だけで算術なし → exact 一致が正しい
    }

    #[test]
    fn test_gpu_double() {
        let gpu = Gpu::new();
        // 64 で割り切れない要素数にして、シェーダの番兵(bounds check)も検証する
        let input: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let output = gpu.double(&input);
        // ×2.0 は IEEE 浮動小数で厳密(指数部 +1)なので exact 一致が正しい
        let expected: Vec<f32> = input.iter().map(|&x| x * 2.0).collect();
        assert_eq!(expected, output);
    }

    #[test]
    fn test_matmul_small_exact() {
        let gpu = Gpu::new();
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];
        let out = gpu.matmul(&a, &b);
        // 整数値の f32 演算は 2^24 まで厳密で、しかも加算順に依存しない → exact が正しい
        let expected = array![[58.0, 64.0], [139.0, 154.0]];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_matmul_vs_ndarray() {
        let gpu = Gpu::new();
        // 16 で割り切れない素数サイズで 2 次元の番兵を検証
        let a: Array2<f32> = Array2::random((37, 53), StandardNormal);
        let b: Array2<f32> = Array2::random((53, 29), StandardNormal);
        let gpu_out = gpu.matmul(&a, &b);
        let cpu_out = a.dot(&b);
        let max_diff = gpu_out
            .iter()
            .zip(cpu_out.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(gpu_out.dim(), cpu_out.dim()); // zip 切り詰め対策
        assert!(max_diff < 1e-3, "max diff {max_diff:e}");
    }

    #[test]
    #[ignore] // ベンチ: cargo test --release -p vol1 bench_matmul -- --ignored --nocapture
    fn bench_matmul_gpu_vs_cpu() {
        let gpu = Gpu::new();
        // (名前, m, k, n): DL 実サイズ 3 つ + スケーリング確認用の正方 3 つ
        let cases = [
            ("conv1_2 im2col", 78400, 144, 16),
            ("conv3_2 im2col", 6400, 576, 64),
            ("affine1", 100, 1024, 50),
            ("square 512", 512, 512, 512),
            ("square 1024", 1024, 1024, 1024),
            ("square 2048", 2048, 2048, 2048),
        ];
        for (name, m, k, n) in cases {
            let a: Array2<f32> = Array2::random((m, k), StandardNormal);
            let b: Array2<f32> = Array2::random((k, n), StandardNormal);
            let flops = 2.0 * m as f64 * k as f64 * n as f64; // 積和 = 2 FLOP

            let iters = 3;
            let _ = a.dot(&b); // ウォームアップ
            let t = Instant::now();
            for _ in 0..iters {
                black_box(a.dot(&b));
            }
            let cpu_s = t.elapsed().as_secs_f64() / iters as f64;

            let _ = gpu.matmul(&a, &b); // ウォームアップ
            let t = Instant::now();
            for _ in 0..iters {
                black_box(gpu.matmul(&a, &b));
            }
            let gpu_s = t.elapsed().as_secs_f64() / iters as f64;

            println!(
                "{name:15} ({m:5},{k:4})x({k:4},{n:4}): CPU {:8.2} ms ({:7.1} GFLOP/s) | GPU {:8.2} ms ({:7.1} GFLOP/s) | GPU/CPU 速度比 x{:.2}",
                cpu_s * 1e3,
                flops / cpu_s / 1e9,
                gpu_s * 1e3,
                flops / gpu_s / 1e9,
                cpu_s / gpu_s
            );
        }
    }

    #[test]
    fn test_matmul_gpu_chain() {
        let gpu = Gpu::new();
        let a: Array2<f32> = Array2::random((37, 53), StandardNormal);
        let b: Array2<f32> = Array2::random((53, 29), StandardNormal);
        let c: Array2<f32> = Array2::random((29, 41), StandardNormal);
        // 中間結果を CPU に降ろさず 2 段連鎖
        let (ga, gb, gc) = (gpu.upload(&a), gpu.upload(&b), gpu.upload(&c));
        let out = gpu.download(&gpu.matmul_gpu(&gpu.matmul_gpu(&ga, &gb), &gc));
        let expected = a.dot(&b).dot(&c);
        let max_diff = out
            .iter()
            .zip(expected.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(out.dim(), expected.dim());
        assert!(max_diff < 1e-3, "max diff {max_diff:e}");
    }

    #[test]
    fn test_relu_gpu() {
        let gpu = Gpu::new();
        let a: Array2<f32> = Array2::random((123, 45), StandardNormal);

        let mut ga = gpu.upload(&a);
        gpu.relu_gpu(&mut ga);
        let out = gpu.download(&ga);

        let expected = a.mapv(|v| v.max(0.0));

        assert_eq!(out, expected);
    }

    #[test]
    fn test_add_bias_gpu() {
        let gpu = Gpu::new();
        let a: Array2<f32> = Array2::random((123, 45), StandardNormal);
        let b: Array2<f32> = Array2::random((1, 45), StandardNormal);

        let mut ga = gpu.upload(&a);
        let gb = gpu.upload(&b);

        gpu.add_bias_gpu(&mut ga, &gb);
        let out = gpu.download(&ga);

        let expected = a + &b;

        assert_eq!(out, expected);
    }

    #[test]
    fn test_affine_forward_gpu() {
        let gpu = Gpu::new();
        let x: Array2<f32> = Array2::random((37, 53), StandardNormal);
        let w: Array2<f32> = Array2::random((53, 29), StandardNormal);
        let b: Array2<f32> = Array2::random((1, 29), StandardNormal);

        let gx = gpu.upload(&x);
        let gw = gpu.upload(&w);
        let gb = gpu.upload(&b);

        let mut out_gpu = gpu.matmul_gpu(&gx, &gw);
        gpu.add_bias_gpu(&mut out_gpu, &gb);
        gpu.relu_gpu(&mut out_gpu);

        let out = gpu.download(&out_gpu);

        let expected = (x.dot(&w) + &b).mapv(|v| v.max(0.0));

        let max_diff = out
            .iter()
            .zip(expected.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);

        assert_eq!(out.dim(), expected.dim());
        assert!(max_diff < 1e-3, "max diff {max_diff:e}");
    }

    #[test]
    #[ignore] // ベンチ: cargo test --release -p vol1 bench_affine_chain -- --ignored --nocapture
    fn bench_affine_chain_resident_vs_roundtrip() {
        let gpu = Gpu::new();
        let layers = 4;
        let dim = 1024;
        for batch in [100usize, 1024] {
            let x: Array2<f32> = Array2::random((batch, dim), StandardNormal);
            let ws: Vec<Array2<f32>> = (0..layers)
                .map(|_| Array2::random((dim, dim), StandardNormal))
                .collect();
            let bs: Vec<Array2<f32>> = (0..layers)
                .map(|_| Array2::random((1, dim), StandardNormal))
                .collect();
            let iters = 3;

            // --- CPU ---
            let cpu_forward = |x: &Array2<f32>| {
                ws.iter()
                    .zip(&bs)
                    .fold(x.clone(), |y, (w, b)| (y.dot(w) + b).mapv(|v| v.max(0.0)))
            };
            let _ = cpu_forward(&x);
            let t = Instant::now();
            for _ in 0..iters {
                black_box(cpu_forward(&x));
            }
            let cpu_s = t.elapsed().as_secs_f64() / iters as f64;

            // --- GPU 毎層往復(naive な dot 置き換え相当) ---
            let _ = gpu.matmul(&x, &ws[0]);
            let t = Instant::now();
            for _ in 0..iters {
                let mut y = x.clone();
                for (w, b) in ws.iter().zip(&bs) {
                    let mut gy = gpu.matmul_gpu(&gpu.upload(&y), &gpu.upload(w));
                    gpu.add_bias_gpu(&mut gy, &gpu.upload(b));
                    gpu.relu_gpu(&mut gy);
                    y = gpu.download(&gy); // 毎層 CPU に読み戻す(ここが無駄)
                }
                black_box(y);
            }
            let roundtrip_s = t.elapsed().as_secs_f64() / iters as f64;

            // --- GPU 常駐(重みはループ外で 1 回だけアップロード) ---
            let gws: Vec<GpuTensor> = ws.iter().map(|w| gpu.upload(w)).collect();
            let gbs: Vec<GpuTensor> = bs.iter().map(|b| gpu.upload(b)).collect();
            let t = Instant::now();
            for _ in 0..iters {
                let mut gy = gpu.upload(&x); // 上りはバッチだけ
                for (gw, gb) in gws.iter().zip(&gbs) {
                    gy = gpu.matmul_gpu(&gy, gw);
                    gpu.add_bias_gpu(&mut gy, gb);
                    gpu.relu_gpu(&mut gy);
                }
                black_box(gpu.download(&gy)); // 下りは最終結果だけ
            }
            let resident_s = t.elapsed().as_secs_f64() / iters as f64;

            let flops = 2.0 * batch as f64 * dim as f64 * dim as f64 * layers as f64;
            println!(
                "batch {batch:4}: CPU {:7.2} ms ({:6.1} GF/s) | GPU往復 {:7.2} ms | GPU常駐 {:7.2} ms | 常駐/CPU x{:.2} | 常駐/往復 x{:.2}",
                cpu_s * 1e3,
                flops / cpu_s / 1e9,
                roundtrip_s * 1e3,
                resident_s * 1e3,
                cpu_s / resident_s,
                roundtrip_s / resident_s,
            );
        }
    }
}
