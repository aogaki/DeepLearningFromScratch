pub mod deep_conv_net;
pub mod layers;

use ndarray::{Array2, Array4};
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

#[derive(Clone)]
pub struct GpuTensor {
    pub buffer: wgpu::Buffer,
    pub shape: (usize, usize),
}

/// GPU 常駐の 4D 画像(NCHW 平坦の GpuTensor + 元の形)
#[derive(Clone)]
pub struct GpuImage {
    pub tensor: GpuTensor,                  // shape = (n, c·h·w)
    pub dims: (usize, usize, usize, usize), // (n, c, h, w)
}

pub struct Gpu {
    pub device: Device,
    pub queue: Queue,
    matmul_pipeline: wgpu::ComputePipeline,
    relu_pipeline: wgpu::ComputePipeline,
    relu_backward_pipeline: wgpu::ComputePipeline,
    bias_add_pipeline: wgpu::ComputePipeline,
    im2col_pipeline: wgpu::ComputePipeline,
    col2im_pipeline: wgpu::ComputePipeline,
    nhwc_to_nchw_pipeline: wgpu::ComputePipeline,
    nchw_to_nhwc_pipeline: wgpu::ComputePipeline,
    pooling_pipeline: wgpu::ComputePipeline,
    pool_backward_pipeline: wgpu::ComputePipeline,
    matmul_tn_pipeline: wgpu::ComputePipeline,
    matmul_nt_pipeline: wgpu::ComputePipeline,
    column_sum_pipeline: wgpu::ComputePipeline,
    sgd_pipeline: wgpu::ComputePipeline,
    pub adam_pipeline: wgpu::ComputePipeline,
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
        let relu_backward_pipeline =
            Self::make_pipeline(&device, include_str!("relu_backward.wgsl"), "relu_backward");
        let bias_add_pipeline =
            Self::make_pipeline(&device, include_str!("bias_add.wgsl"), "bias_add");
        let im2col_pipeline = Self::make_pipeline(&device, include_str!("im2col.wgsl"), "im2col");
        let col2im_pipeline = Self::make_pipeline(&device, include_str!("col2im.wgsl"), "col2im");
        let nhwc_to_nchw_pipeline =
            Self::make_pipeline(&device, include_str!("nhwc_to_nchw.wgsl"), "nhwc_to_nchw");
        let nchw_to_nhwc_pipeline =
            Self::make_pipeline(&device, include_str!("nchw_to_nhwc.wgsl"), "nchw_to_nhwc");
        let pooling_pipeline =
            Self::make_pipeline(&device, include_str!("pooling.wgsl"), "pooling");
        let pool_backward_pipeline =
            Self::make_pipeline(&device, include_str!("pool_backward.wgsl"), "pool_backward");
        let matmul_tn_pipeline =
            Self::make_pipeline(&device, include_str!("matmul_tn.wgsl"), "matmul_tn");
        let column_sum_pipeline =
            Self::make_pipeline(&device, include_str!("column_sum.wgsl"), "column_sum");
        let matmul_nt_pipeline =
            Self::make_pipeline(&device, include_str!("matmul_nt.wgsl"), "matmul_nt");
        let sgd_pipeline = Self::make_pipeline(&device, include_str!("sgd.wgsl"), "sgd");
        let adam_pipeline = Self::make_pipeline(&device, include_str!("adam.wgsl"), "adam");

        Gpu {
            device,
            queue,
            matmul_pipeline,
            relu_pipeline,
            relu_backward_pipeline,
            bias_add_pipeline,
            im2col_pipeline,
            col2im_pipeline,
            nhwc_to_nchw_pipeline,
            nchw_to_nhwc_pipeline,
            pooling_pipeline,
            pool_backward_pipeline,
            matmul_tn_pipeline,
            matmul_nt_pipeline,
            column_sum_pipeline,
            sgd_pipeline,
            adam_pipeline,
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

    /// C (m,n) = Aᵀ·B。A は (k,m) 格納のまま転置読みする(実体化しない)
    pub fn matmul_tn_gpu(&self, a: &GpuTensor, b: &GpuTensor) -> GpuTensor {
        let (k, m) = a.shape;
        let (k2, n) = b.shape;
        assert_eq!(k, k2, "matmul_tn_gpu: sum dimensions must match");
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
            layout: &self.matmul_tn_pipeline.get_bind_group_layout(0),
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
            pass.set_pipeline(&self.matmul_tn_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(m.div_ceil(4) as u32, n.div_ceil(4) as u32, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        GpuTensor {
            buffer: out_buf,
            shape: (m, n),
        }
    }

    /// C(m,n) = A(m,k) · B(n,k)ᵀ
    /// a と b 共に k 方向が連続であることを利用し、転置を事前実体化せずに計算する
    pub fn matmul_nt_gpu(&self, a: &GpuTensor, b: &GpuTensor) -> GpuTensor {
        let (m, k) = a.shape;
        let (n, k2) = b.shape;
        assert_eq!(k, k2, "matmul_nt_gpu: k dimensions must match");

        let dims: [u32; 4] = [m as u32, k as u32, n as u32, 0];
        let dims_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matmul_nt dims"),
                contents: bytemuck::cast_slice(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matmul_nt out"),
            size: (m * n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let block_x = 16;
        let block_y = 16;
        let grid_x = (n as u32).div_ceil(block_x);
        let grid_y = (m as u32).div_ceil(block_y);

        // dispatch_2d ヘルパーを呼び出す
        self.dispatch_2d(
            &self.matmul_nt_pipeline,
            &[
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
            (grid_x, grid_y),
        );

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

    /// 2 次元 dispatch の定型(dispatch_1d の (x,y) 版。groups は workgroup 数)
    fn dispatch_2d(
        &self,
        pipeline: &wgpu::ComputePipeline,
        entries: &[wgpu::BindGroupEntry],
        groups: (u32, u32),
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
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Array4 画像を (n, c·h·w) に平坦化して GPU へ
    pub fn upload_image(&self, x: &Array4<f32>) -> GpuImage {
        let (n, c, h, w) = x.dim();
        let flat = x.as_standard_layout();
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("image"),
                contents: bytemuck::cast_slice(flat.as_slice().expect("standard layout")),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        GpuImage {
            tensor: GpuTensor {
                buffer,
                shape: (n, c * h * w),
            },
            dims: (n, c, h, w),
        }
    }

    /// GPU 常駐 im2col: x は (n, c·h·w) 平坦画像、出力は (n·oh·ow, c·fh·fw)
    pub fn im2col_gpu(
        &self,
        x: &GpuImage,
        fh: usize,
        fw: usize,
        stride: usize,
        pad: usize,
    ) -> GpuTensor {
        let (n, c, h, w) = x.dims;
        assert_eq!(x.tensor.shape, (n, c * h * w), "image shape mismatch");
        let oh = (h + 2 * pad - fh) / stride + 1;
        let ow = (w + 2 * pad - fw) / stride + 1;
        let (rows, cols) = (n * oh * ow, c * fh * fw);

        let params: [u32; 12] = [
            n as u32,
            c as u32,
            h as u32,
            w as u32,
            oh as u32,
            ow as u32,
            fh as u32,
            fw as u32,
            stride as u32,
            pad as u32,
            0,
            0,
        ];
        let p_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("im2col params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let out = GpuTensor {
            buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("col"),
                size: (rows * cols * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            shape: (rows, cols),
        };
        self.dispatch_2d(
            &self.im2col_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: p_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x.tensor.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer.as_entire_binding(),
                },
            ],
            (cols.div_ceil(16) as u32, rows.div_ceil(16) as u32),
        );
        out
    }

    /// dcol (n·oh·ow, c·fh·fw) → dx 画像。gather 型(atomic 不要・決定論)
    pub fn col2im_gpu(
        &self,
        dcol: &GpuTensor,
        input_dims: (usize, usize, usize, usize),
        fh: usize,
        fw: usize,
        stride: usize,
        pad: usize,
    ) -> GpuImage {
        let (n, c, h, w) = input_dims;
        let oh = (h + 2 * pad - fh) / stride + 1;
        let ow = (w + 2 * pad - fw) / stride + 1;

        assert_eq!(
            dcol.shape,
            (n * oh * ow, c * fh * fw),
            "col2im_gpu: dcol shape mismatch"
        );

        // uniformアライメント(16byteの倍数)を満たすためパディングして12要素(48byte)
        let dims: [u32; 12] = [
            n as u32,
            c as u32,
            h as u32,
            w as u32,
            oh as u32,
            ow as u32,
            fh as u32,
            fw as u32,
            stride as u32,
            pad as u32,
            0,
            0,
        ];

        let dims_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("col2im dims"),
                contents: bytemuck::cast_slice(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("col2im out (dx)"),
            size: (n * c * h * w * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.dispatch_1d(
            &self.col2im_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dims_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dcol.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
            n * c * h * w, // 1 thread = dx の 1 要素
        );

        GpuImage {
            tensor: GpuTensor {
                buffer: out_buf,
                shape: (n, c * h * w),
            },
            dims: (n, c, h, w),
        }
    }

    /// (n·oh·ow, f) → NCHW 平坦 (n, f·oh·ow)。conv を conv に繋ぐための並べ替え
    pub fn nhwc_to_nchw_gpu(
        &self,
        src: &GpuTensor,
        dims: (usize, usize, usize, usize),
    ) -> GpuImage {
        let (n, f, oh, ow) = dims;
        assert_eq!(src.shape, (n * oh * ow, f));
        let total = n * f * oh * ow;
        let params: [u32; 4] = [n as u32, f as u32, oh as u32, ow as u32];
        let p_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nhwc_to_nchw params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let dst = GpuTensor {
            buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nhwc_to_nchw dst"),
                size: (total * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            shape: (n, f * oh * ow),
        };
        self.dispatch_1d(
            &self.nhwc_to_nchw_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: p_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dst.buffer.as_entire_binding(),
                },
            ],
            total,
        );
        GpuImage {
            tensor: dst,
            dims: (n, f, oh, ow),
        }
    }

    /// NCHW 画像 → (n·oh·ow, f) 行列 (nhwc_to_nchw の逆)
    pub fn nchw_to_nhwc_gpu(&self, src: &GpuImage) -> GpuTensor {
        let (n, f, oh, ow) = src.dims; // conv dout なので C は filter 数 (f)
        let dims: [u32; 4] = [n as u32, f as u32, oh as u32, ow as u32];

        let dims_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nchw_to_nhwc dims"),
                contents: bytemuck::cast_slice(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dout2d"),
            size: (n * f * oh * ow * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.dispatch_1d(
            &self.nchw_to_nhwc_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dims_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src.tensor.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
            n * f * oh * ow,
        );

        GpuTensor {
            buffer: out_buf,
            shape: (n * oh * ow, f),
        }
    }

    /// GPU 常駐 conv forward: im2col → matmul → bias → NCHW 並べ替え。
    /// w_colt は w.reshape(fn, c·fh·fw) の転置 (c·fh·fw, fn)、bias は (1, fn)(CPU で準備して upload)
    pub fn conv_forward_gpu(
        &self,
        x: &GpuImage,
        w_colt: &GpuTensor,
        bias: &GpuTensor,
        fh: usize,
        fw: usize,
        stride: usize,
        pad: usize,
    ) -> GpuImage {
        let (n, _c, h, w) = x.dims;
        let fn_ = w_colt.shape.1;
        let oh = (h + 2 * pad - fh) / stride + 1;
        let ow = (w + 2 * pad - fw) / stride + 1;
        let col = self.im2col_gpu(&x, fh, fw, stride, pad);
        let mut y = self.matmul_gpu(&col, w_colt);
        self.add_bias_gpu(&mut y, bias);
        self.nhwc_to_nchw_gpu(&y, (n, fn_, oh, ow))
    }

    /// conv backward: dout 画像 + forward の col から (dx, dW_colt, db) を計算。
    pub fn conv_backward_gpu(
        &self,
        dout: &GpuImage,
        col: &GpuTensor,
        w_colt: &GpuTensor,
        input_dims: (usize, usize, usize, usize),
        fh: usize,
        fw: usize,
        stride: usize,
        pad: usize,
    ) -> (GpuImage, GpuTensor, GpuTensor) {
        let dout2d = self.nchw_to_nhwc_gpu(dout);

        let dw_colt = self.matmul_tn_gpu(col, &dout2d);

        let db = self.column_sum_gpu(&dout2d);

        let dcol = self.matmul_nt_gpu(&dout2d, w_colt);

        let dx = self.col2im_gpu(&dcol, input_dims, fh, fw, stride, pad);

        (dx, dw_colt, db)
    }

    pub fn pool_forward_gpu(
        &self,
        x: &GpuImage,
        ph: usize,
        pw: usize,
        stride: usize,
        pad: usize,
    ) -> (GpuImage, wgpu::Buffer) {
        let (n, c, h, w) = x.dims;
        let oh = (h + 2 * pad - ph) / stride + 1;
        let ow = (w + 2 * pad - pw) / stride + 1;
        let total = n * c * oh * ow;

        // uniform パラメータ
        let params: [u32; 12] = [
            n as u32,
            c as u32,
            h as u32,
            w as u32,
            oh as u32,
            ow as u32,
            ph as u32,
            pw as u32,
            stride as u32,
            pad as u32,
            0,
            0,
        ];
        let p_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pooling params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // 出力画像用バッファ (f32)
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pooling out"),
            size: (total * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // argmax 用バッファ (u32)
        // テスト等で覗く可能性を見越して COPY_SRC を付与。
        // u32 の読み出し関数が整備されるまでは生の wgpu::Buffer として安全に保持する。
        let argmax_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pooling argmax"),
            size: (total * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.dispatch_1d(
            &self.pooling_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: p_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x.tensor.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: argmax_buf.as_entire_binding(),
                },
            ],
            total,
        );

        let out_img = GpuImage {
            tensor: GpuTensor {
                buffer: out_buf,
                shape: (n, c * oh * ow),
            },
            dims: (n, c, oh, ow), // チャンネル数(C)は畳み込みと違い入力のまま維持される
        };

        (out_img, argmax_buf)
    }

    pub fn pool_backward_gpu(
        &self,
        dout: &GpuImage,
        argmax: &wgpu::Buffer,
        x_dims: (usize, usize, usize, usize),
        ph: usize,
        pw: usize,
        stride: usize,
        pad: usize,
    ) -> GpuImage {
        // 将来 overlap を使おうとした自分への警告文
        assert!(
            stride >= ph.max(pw),
            "overlapping pooling requires atomic scatter"
        );

        let (n, c, oh, ow) = dout.dims;
        let (_, _, h, w) = x_dims;

        // uniformアライメント(16byteの倍数)を満たすため、末尾をパディングして12要素(48byte)にする
        let dims: [u32; 12] = [
            n as u32,
            c as u32,
            h as u32,
            w as u32,
            oh as u32,
            ow as u32,
            ph as u32,
            pw as u32,
            stride as u32,
            pad as u32,
            0,
            0,
        ];

        let dims_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pool_backward dims"),
                contents: bytemuck::cast_slice(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // WebGPU仕様によるゼロ初期化を利用（明示的クリア不要）
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pool_backward out"),
            size: (n * c * h * w * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.dispatch_1d(
            &self.pool_backward_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dims_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dout.tensor.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: argmax.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
            n * c * oh * ow,
        );

        GpuImage {
            tensor: GpuTensor {
                buffer: out_buf,
                shape: (n, c * h * w),
            },
            dims: (n, c, h, w),
        }
    }

    /// 列ごとの総和 (rows, cols) → (1, cols)。affine/conv の db 用
    pub fn column_sum_gpu(&self, x: &GpuTensor) -> GpuTensor {
        let (rows, cols) = x.shape;
        let dims: [u32; 4] = [rows as u32, cols as u32, 0, 0];
        let dims_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("column_sum dims"),
                contents: bytemuck::cast_slice(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("column_sum out"),
            size: (cols * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.column_sum_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dims_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.column_sum_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // cols 列に対して cols 個のワークグループを起動 (1グループ256人体制)
            pass.dispatch_workgroups(cols as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        GpuTensor {
            buffer: out_buf,
            shape: (1, cols),
        }
    }

    /// ReLU backward (in-place)
    /// dout = dout * (act > 0.0)
    pub fn relu_backward_gpu(&self, dout: &mut GpuTensor, act: &GpuTensor) {
        assert_eq!(dout.shape, act.shape, "relu_backward_gpu: shape mismatch");
        let n = dout.shape.0 * dout.shape.1;
        self.dispatch_1d(
            &self.relu_backward_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dout.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: act.buffer.as_entire_binding(),
                },
            ],
            n,
        );
    }

    /// SGD (param = param - lr * grad)
    /// 要素ごとの in-place 更新
    pub fn sgd_update_gpu(&self, param: &mut GpuTensor, grad: &GpuTensor, lr: f32) {
        // バインドグループは形状不一致でも黙って通ってしまうため、ここで防ぐ
        assert_eq!(param.shape, grad.shape, "sgd_update_gpu: shape mismatch");
        let n = param.shape.0 * param.shape.1;

        // 4バイトの f32 スカラーをそのまま uniform バッファとして確保
        let lr_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sgd lr"),
                contents: bytemuck::cast_slice(&[lr]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        self.dispatch_1d(
            &self.sgd_pipeline, // ※ new() 内での make_pipeline 登録前提
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lr_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: param.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grad.buffer.as_entire_binding(),
                },
            ],
            n,
        );
    }
}

pub struct GpuAdam {
    iter: i32,
    m: Option<GpuTensor>,
    v: Option<GpuTensor>,
}

impl GpuAdam {
    pub fn new() -> Self {
        Self {
            iter: 0,
            m: None,
            v: None,
        }
    }

    pub fn update(&mut self, gpu: &Gpu, param: &mut GpuTensor, grad: &GpuTensor, lr: f32) {
        use wgpu::util::DeviceExt;
        self.iter += 1;
        let elements = param.shape.0 * param.shape.1;

        let m = self.m.get_or_insert_with(|| {
            let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("adam_m"),
                size: (elements * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            GpuTensor { buffer, shape: param.shape }
        });

        let v = self.v.get_or_insert_with(|| {
            let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("adam_v"),
                size: (elements * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            GpuTensor { buffer, shape: param.shape }
        });

        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let c1 = 1.0 / (1.0 - beta1.powi(self.iter));
        let c2 = 1.0 / (1.0 - beta2.powi(self.iter));

        let uniforms = [lr, c1, c2, 0.0f32];
        let uniform_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("adam_uniforms"),
            contents: bytemuck::cast_slice(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group_layout = gpu.adam_pipeline.get_bind_group_layout(0);
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("adam_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: param.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grad.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: m.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: v.buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&gpu.adam_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(elements.div_ceil(64) as u32, 1, 1);
        }
        gpu.queue.submit(Some(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conv::{ConvolutionLayer, PoolingLayer};
    use crate::layers::{AffineLayer, FlattenLayer, Layer, ReluLayer};
    use ndarray::{Array1, Array2, Array4, Ix2, array};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::StandardNormal;
    use std::assert_eq;
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

    use crate::conv::im2col;
    #[test]
    fn test_im2col_gpu() {
        let gpu = Gpu::new();
        for (n, c, h, w, fh, fw, stride, pad) in [
            (
                1usize, 3usize, 7usize, 7usize, 5usize, 5usize, 1usize, 0usize,
            ), // 本 7.4.3 の形
            (2, 2, 4, 4, 2, 2, 2, 0),    // stride 2
            (2, 3, 5, 6, 3, 3, 1, 1),    // 非正方 + pad
            (1, 1, 4, 4, 3, 3, 1, 2),    // pad=2(DeepConvNet conv4 と同じ味)
            (2, 16, 28, 28, 3, 3, 1, 1), // conv1_2 実形(N は縮小)
        ] {
            let x: Array4<f32> = Array4::random((n, c, h, w), StandardNormal);
            let gcol = gpu.im2col_gpu(&gpu.upload_image(&x), fh, fw, stride, pad);
            // im2col は移動のみ・算術ゼロ → exact 一致(ch7 のルールの GPU 版)
            assert_eq!(gpu.download(&gcol), im2col(&x, fh, fw, stride, pad));
        }
    }

    #[test]
    fn test_conv_forward_gpu() {
        let gpu = Gpu::new();
        // (n, c, h, w, fn_, fh, fw, stride, pad)
        for (n, c, h, w, fn_, fh, fw, stride, pad) in [
            (2, 3, 7, 7, 5, 3, 3, 1, 1), // プローブ層の形 (pad 1)
            (2, 3, 7, 7, 5, 3, 3, 2, 2), // Stride 2 と Pad 2 のテスト
        ] {
            let x: Array4<f32> = Array4::random((n, c, h, w), StandardNormal);
            let weight: Array4<f32> = Array4::random((fn_, c, fh, fw), StandardNormal);
            let bias = ndarray::Array1::random(fn_, StandardNormal);

            // CPU
            let mut conv = ConvolutionLayer::new(weight.clone(), bias.clone(), stride, pad);
            let cpu_out = conv.forward(&x); // Array4<f32> が返る

            // GPU
            // w_colt: (c*fh*fw, fn) への変形と転置 (標準レイアウト化のために assign を使用)
            let mut w_colt = Array2::<f32>::zeros((c * fh * fw, fn_));
            w_colt.assign(
                &weight
                    .into_shape_with_order((fn_, c * fh * fw))
                    .unwrap()
                    .reversed_axes(),
            );

            let gx = gpu.upload_image(&x);
            let gw = gpu.upload(&w_colt);
            let gb = gpu.upload(&bias.into_shape_with_order((1, fn_)).unwrap());

            let gy = gpu.conv_forward_gpu(&gx, &gw, &gb, fh, fw, stride, pad);
            let gpu_out = gpu.download(&gy.tensor);

            // 比較
            assert_eq!(
                gy.dims,
                (
                    n,
                    fn_,
                    (h + 2 * pad - fh) / stride + 1,
                    (w + 2 * pad - fw) / stride + 1
                )
            );
            // GPU の出力は NCHW 平坦化された 2D (n, fn*oh*ow) なので、CPU 側もそれに合わせて平坦化
            let cpu_flat = cpu_out
                .as_standard_layout()
                .into_owned()
                .into_shape_with_order(gpu_out.dim())
                .unwrap();

            let max_diff = gpu_out
                .iter()
                .zip(cpu_flat.iter())
                .map(|(g, c)| (g - c).abs())
                .fold(0.0f32, f32::max);

            assert!(max_diff < 1e-3, "max diff {max_diff:e}");
        }
    }

    #[test]
    fn test_conv_chain_gpu() {
        let gpu = Gpu::new();
        let (n, c, h, w) = (2, 3, 10, 10);

        let (fn1, fh1, fw1) = (4, 3, 3);
        let (fn2, fh2, fw2) = (5, 3, 3);

        let x: Array4<f32> = Array4::random((n, c, h, w), StandardNormal);

        let w1: Array4<f32> = Array4::random((fn1, c, fh1, fw1), StandardNormal);
        let b1 = ndarray::Array1::random(fn1, StandardNormal);

        let w2: Array4<f32> = Array4::random((fn2, fn1, fh2, fw2), StandardNormal);
        let b2 = ndarray::Array1::random(fn2, StandardNormal);

        // --- CPU 側の連鎖 ---
        let mut conv1 = ConvolutionLayer::new(w1.clone(), b1.clone(), 1, 1);
        let mut conv2 = ConvolutionLayer::new(w2.clone(), b2.clone(), 1, 1);

        let c1_out = conv1.forward(&x);
        let relu_out = c1_out.mapv(|v| v.max(0.0));
        let cpu_expected = conv2.forward(&relu_out);

        // --- GPU 側の連鎖 ---
        let gx = gpu.upload_image(&x);

        // 重み 1
        let mut w1_colt = Array2::<f32>::zeros((c * fh1 * fw1, fn1));
        w1_colt.assign(
            &w1.into_shape_with_order((fn1, c * fh1 * fw1))
                .unwrap()
                .reversed_axes(),
        );
        let gw1 = gpu.upload(&w1_colt);
        let gb1 = gpu.upload(&b1.into_shape_with_order((1, fn1)).unwrap());

        // 重み 2
        let mut w2_colt = Array2::<f32>::zeros((fn1 * fh2 * fw2, fn2));
        w2_colt.assign(
            &w2.into_shape_with_order((fn2, fn1 * fh2 * fw2))
                .unwrap()
                .reversed_axes(),
        );
        let gw2 = gpu.upload(&w2_colt);
        let gb2 = gpu.upload(&b2.into_shape_with_order((1, fn2)).unwrap());

        // GPU での実行: conv1 -> ReLU -> conv2
        let mut gy1 = gpu.conv_forward_gpu(&gx, &gw1, &gb1, fh1, fw1, 1, 1);

        // relu_gpu は GpuTensor への要素単位演算のため、GpuImage(.tensor) の形状を変えずにそのまま適用可能
        gpu.relu_gpu(&mut gy1.tensor);

        // NCHW に並べ替えられた gy1 を、そのまま次の conv_forward_gpu に渡す
        let gy2 = gpu.conv_forward_gpu(&gy1, &gw2, &gb2, fh2, fw2, 1, 1);

        let gpu_out = gpu.download(&gy2.tensor);

        // --- 比較 ---
        assert_eq!(
            gy2.dims,
            (n, fn2, (h + 2 * 1 - fh2) / 1 + 1, (w + 2 * 1 - fw2) / 1 + 1)
        );
        let cpu_flat = cpu_expected
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order(gpu_out.dim())
            .unwrap();

        let max_diff = gpu_out
            .iter()
            .zip(cpu_flat.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);

        assert!(max_diff < 1e-3, "max diff {max_diff:e}");
    }

    #[test]
    fn test_pool_forward_gpu() {
        let gpu = Gpu::new();

        // パラメータ: (n, c, h, w, pool_h, pool_w, stride)
        for (n, c, h, w, ph, pw, stride) in [
            (2, 3, 4, 4, 2, 2, 2), // DeepConvNet実形: 2x2 stride 2 (N≥2, C≥2)
            (2, 2, 7, 7, 2, 2, 2), // 奇数サイズ: 7x7 -> 3x3 に切り捨てられるケース
            (3, 4, 6, 6, 3, 3, 1), // 別の形状パターン
        ] {
            let pad = 0; // exact 一致検証のため pad は 0 固定
            let x: Array4<f32> = Array4::random((n, c, h, w), StandardNormal);

            // --- CPU 側 ---
            let mut pool = PoolingLayer::new(ph, pw, stride, pad);
            let cpu_out = pool.forward(&x); // 戻り値は (n, c, oh, ow) だがメモリは NHWC
            let (_, _, oh, ow) = cpu_out.dim();

            // --- GPU 側 ---
            let gx = gpu.upload_image(&x);
            let (gy, _argmax) = gpu.pool_forward_gpu(&gx, ph, pw, stride, pad);
            let gpu_out = gpu.download(&gy.tensor); // 戻り値は (n, c*oh*ow)

            // --- 比較 ---
            // CPU 側の Array4 は内部が非連続メモリになっているため、標準レイアウトに直してから NCHW 平坦化
            let cpu_flat = cpu_out
                .as_standard_layout()
                .into_owned()
                .into_shape_with_order((n, c * oh * ow))
                .unwrap();

            // Max は純粋な選択操作（算術演算なし）なので exact に一致する
            assert_eq!(gpu_out, cpu_flat);
        }
    }

    fn run_deep_conv_net_gpu_vs_cpu(batch: usize) {
        let gpu = Gpu::new();
        let x: Array4<f32> = Array4::random((batch, 1, 28, 28), StandardNormal);

        let he_conv = |fn_: usize, c: usize, fh: usize, fw: usize| -> Array4<f32> {
            let fan_in = (c * fh * fw) as f32;
            let scale = (2.0 / fan_in).sqrt();
            Array4::random((fn_, c, fh, fw), StandardNormal) * scale
        };
        let he_affine = |fan_in: usize, fan_out: usize| -> Array2<f32> {
            let scale = (2.0 / fan_in as f32).sqrt();
            Array2::random((fan_in, fan_out), StandardNormal) * scale
        };

        // Weights and biases for both CPU and GPU
        let w1_1 = he_conv(16, 1, 3, 3);
        let b1_1 = Array1::<f32>::zeros(16);
        let w1_2 = he_conv(16, 16, 3, 3);
        let b1_2 = Array1::<f32>::zeros(16);

        let w2_1 = he_conv(32, 16, 3, 3);
        let b2_1 = Array1::<f32>::zeros(32);
        let w2_2 = he_conv(32, 32, 3, 3); // 後の ConvolutionLayer 構築で pad=2 を指定
        let b2_2 = Array1::<f32>::zeros(32);

        let w3_1 = he_conv(64, 32, 3, 3);
        let b3_1 = Array1::<f32>::zeros(64);
        let w3_2 = he_conv(64, 64, 3, 3);
        let b3_2 = Array1::<f32>::zeros(64);

        let wa1 = he_affine(64 * 4 * 4, 50);
        let ba1 = Array2::<f32>::zeros((1, 50));
        let wa2 = he_affine(50, 10);
        let ba2 = Array2::<f32>::zeros((1, 10));

        // CPU
        let mut c1_1 = ConvolutionLayer::new(w1_1.clone(), b1_1.clone(), 1, 1);
        let mut r1_1 = ReluLayer::new();
        let mut c1_2 = ConvolutionLayer::new(w1_2.clone(), b1_2.clone(), 1, 1);
        let mut r1_2 = ReluLayer::new();
        let mut p1 = PoolingLayer::new(2, 2, 2, 0);

        let mut c2_1 = ConvolutionLayer::new(w2_1.clone(), b2_1.clone(), 1, 1);
        let mut r2_1 = ReluLayer::new();
        let mut c2_2 = ConvolutionLayer::new(w2_2.clone(), b2_2.clone(), 1, 2); // ★ pad=2
        let mut r2_2 = ReluLayer::new();
        let mut p2 = PoolingLayer::new(2, 2, 2, 0);

        let mut c3_1 = ConvolutionLayer::new(w3_1.clone(), b3_1.clone(), 1, 1);
        let mut r3_1 = ReluLayer::new();
        let mut c3_2 = ConvolutionLayer::new(w3_2.clone(), b3_2.clone(), 1, 1);
        let mut r3_2 = ReluLayer::new();
        let mut p3 = PoolingLayer::new(2, 2, 2, 0);

        let mut flat = FlattenLayer::new();
        let mut af1 = AffineLayer::new(wa1.clone(), ba1.clone());
        let mut ra1 = ReluLayer::new();
        let mut af2 = AffineLayer::new(wa2.clone(), ba2.clone());

        let mut cpu_forward = |x: &Array4<f32>| -> Array2<f32> {
            let mut out = x.clone().into_dyn();
            out = Layer::forward(&mut c1_1, out, false);
            out = Layer::forward(&mut r1_1, out, false);
            out = Layer::forward(&mut c1_2, out, false);
            out = Layer::forward(&mut r1_2, out, false);
            out = Layer::forward(&mut p1, out, false);

            out = Layer::forward(&mut c2_1, out, false);
            out = Layer::forward(&mut r2_1, out, false);
            out = Layer::forward(&mut c2_2, out, false);
            out = Layer::forward(&mut r2_2, out, false);
            out = Layer::forward(&mut p2, out, false);

            out = Layer::forward(&mut c3_1, out, false);
            out = Layer::forward(&mut r3_1, out, false);
            out = Layer::forward(&mut c3_2, out, false);
            out = Layer::forward(&mut r3_2, out, false);
            out = Layer::forward(&mut p3, out, false);

            out = Layer::forward(&mut flat, out, false);
            out = Layer::forward(&mut af1, out, false);
            out = Layer::forward(&mut ra1, out, false);
            out = Layer::forward(&mut af2, out, false);
            out.into_dimensionality::<Ix2>().unwrap()
        };

        let _ = cpu_forward(&x); // warmup
        let t_cpu = Instant::now();
        let cpu_out = cpu_forward(&x);
        let cpu_time = t_cpu.elapsed();

        // GPU
        fn prep_conv_w(w: &Array4<f32>) -> Array2<f32> {
            let (fn_, c, fh, fw) = w.dim();
            let mut w_colt = Array2::<f32>::zeros((c * fh * fw, fn_));
            w_colt.assign(
                &w.clone()
                    .into_shape_with_order((fn_, c * fh * fw))
                    .unwrap()
                    .reversed_axes(),
            );
            w_colt
        }

        let gw1_1 = gpu.upload(&prep_conv_w(&w1_1));
        let gb1_1 = gpu.upload(&b1_1.into_shape_with_order((1, 16)).unwrap());
        let gw1_2 = gpu.upload(&prep_conv_w(&w1_2));
        let gb1_2 = gpu.upload(&b1_2.into_shape_with_order((1, 16)).unwrap());

        let gw2_1 = gpu.upload(&prep_conv_w(&w2_1));
        let gb2_1 = gpu.upload(&b2_1.into_shape_with_order((1, 32)).unwrap());
        let gw2_2 = gpu.upload(&prep_conv_w(&w2_2));
        let gb2_2 = gpu.upload(&b2_2.into_shape_with_order((1, 32)).unwrap());

        let gw3_1 = gpu.upload(&prep_conv_w(&w3_1));
        let gb3_1 = gpu.upload(&b3_1.into_shape_with_order((1, 64)).unwrap());
        let gw3_2 = gpu.upload(&prep_conv_w(&w3_2));
        let gb3_2 = gpu.upload(&b3_2.into_shape_with_order((1, 64)).unwrap());

        let gwa1 = gpu.upload(&wa1);
        let gba1 = gpu.upload(&ba1);
        let gwa2 = gpu.upload(&wa2);
        let gba2 = gpu.upload(&ba2);

        let gpu_forward = |x: &Array4<f32>| -> Array2<f32> {
            let mut gx = gpu.upload_image(x);

            // Block 1
            gx = gpu.conv_forward_gpu(&gx, &gw1_1, &gb1_1, 3, 3, 1, 1);
            gpu.relu_gpu(&mut gx.tensor);
            gx = gpu.conv_forward_gpu(&gx, &gw1_2, &gb1_2, 3, 3, 1, 1);
            gpu.relu_gpu(&mut gx.tensor);
            let (gx_pool1, _) = gpu.pool_forward_gpu(&gx, 2, 2, 2, 0);

            // Block 2
            gx = gpu.conv_forward_gpu(&gx_pool1, &gw2_1, &gb2_1, 3, 3, 1, 1);
            gpu.relu_gpu(&mut gx.tensor);
            gx = gpu.conv_forward_gpu(&gx, &gw2_2, &gb2_2, 3, 3, 1, 2); // ★ pad=2
            gpu.relu_gpu(&mut gx.tensor);
            let (gx_pool2, _) = gpu.pool_forward_gpu(&gx, 2, 2, 2, 0);

            // Block 3
            gx = gpu.conv_forward_gpu(&gx_pool2, &gw3_1, &gb3_1, 3, 3, 1, 1);
            gpu.relu_gpu(&mut gx.tensor);
            gx = gpu.conv_forward_gpu(&gx, &gw3_2, &gb3_2, 3, 3, 1, 1);
            gpu.relu_gpu(&mut gx.tensor);
            let (gx_pool3, _) = gpu.pool_forward_gpu(&gx, 2, 2, 2, 0);

            let mut g_af1 = gpu.matmul_gpu(&gx_pool3.tensor, &gwa1);
            gpu.add_bias_gpu(&mut g_af1, &gba1);
            gpu.relu_gpu(&mut g_af1);

            let mut g_af2 = gpu.matmul_gpu(&g_af1, &gwa2);
            gpu.add_bias_gpu(&mut g_af2, &gba2);

            gpu.download(&g_af2)
        };

        let _ = gpu_forward(&x); // warmup
        let t_gpu = Instant::now();
        let gpu_out = gpu_forward(&x);
        let gpu_time = t_gpu.elapsed();

        println!(
            "DeepConvNet Forward (batch={}): CPU {:.2} ms | GPU {:.2} ms | Speedup: {:.2}x",
            batch,
            cpu_time.as_secs_f64() * 1000.0,
            gpu_time.as_secs_f64() * 1000.0,
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
        );

        // 比較対象はsoftmax前のロジット10列
        let max_diff = cpu_out
            .iter()
            .zip(gpu_out.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);

        println!("Max logit diff: {}", max_diff);

        // conv 6段 + affine 2段を経由した誤差蓄積のため、eps = 1e-2 で安全に判定
        assert!(max_diff < 1e-2, "Max diff is too large: {}", max_diff);
    }

    #[test]
    fn test_deepconv_forward_gpu_correctness() {
        run_deep_conv_net_gpu_vs_cpu(2);
    }

    #[test]
    #[ignore] // ベンチ cargo test --release -p vol1 bench_deepconv -- --ignored --nocapture
    fn bench_deepconv_forward_gpu_vs_cpu() {
        run_deep_conv_net_gpu_vs_cpu(100);
    }

    #[test]
    fn test_matmul_tn_gpu() {
        let gpu = Gpu::new();

        // (k, m, n) の組み合わせ
        for (k, m, n) in [
            (3, 7, 5),       // 素数サイズ
            (1000, 100, 50), // 誤差蓄積ケース
        ] {
            // A: (k, m) -> 転置して (m, k) 扱い
            let a: Array2<f32> = Array2::random((k, m), StandardNormal);
            // B: (k, n)
            let b: Array2<f32> = Array2::random((k, n), StandardNormal);

            // CPU: A^T * B -> (m, n)
            let cpu_out = a.t().dot(&b);

            let ga = gpu.upload(&a);
            let gb = gpu.upload(&b);
            let gy = gpu.matmul_tn_gpu(&ga, &gb);
            let gpu_out = gpu.download(&gy);

            let max_diff = cpu_out
                .iter()
                .zip(gpu_out.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max);

            // 誤差蓄積が大きいケースでも 1e-3 以下に収まるか
            assert!(
                max_diff < 1e-3,
                "max diff {max_diff:e} for (k={k}, m={m}, n={n})"
            );
        }
    }

    #[test]
    fn test_column_sum_gpu() {
        use ndarray::Axis;
        let gpu = Gpu::new();

        for (rows, cols) in [(3, 7), (100, 50), (1000, 7)] {
            let x: Array2<f32> = Array2::random((rows, cols), StandardNormal);
            let cpu_out = x
                .sum_axis(Axis(0))
                .into_shape_with_order((1, cols))
                .unwrap();

            let gx = gpu.upload(&x);
            let gy = gpu.column_sum_gpu(&gx);
            let gpu_out = gpu.download(&gy);

            let max_diff = cpu_out
                .iter()
                .zip(gpu_out.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max);

            assert!(max_diff < 1e-4, "max diff {max_diff:e} for ({rows}x{cols})");
        }
    }

    #[test]
    fn test_affine_backward_gpu() {
        use crate::layers::{AffineLayer, Layer};
        use ndarray::Ix2;

        let gpu = Gpu::new();

        let batch = 100;
        let in_size = 50;
        let out_size = 10;

        let x: Array2<f32> = Array2::random((batch, in_size), StandardNormal);
        let w: Array2<f32> = Array2::random((in_size, out_size), StandardNormal);
        let b: Array2<f32> = Array2::zeros((1, out_size));
        let dout: Array2<f32> = Array2::random((batch, out_size), StandardNormal);

        // --- CPU: forward -> backward ---
        let mut affine = AffineLayer::new(w.clone(), b.clone());
        let _ = Layer::forward(&mut affine, x.clone().into_dyn(), false); // xを保持させる
        let cpu_dx = Layer::backward(&mut affine, dout.clone().into_dyn())
            .into_dimensionality::<Ix2>()
            .unwrap();
        let cpu_dw = affine.dw();
        let cpu_db = affine.db();

        // --- GPU ---
        let gx = gpu.upload(&x);
        let gdout = gpu.upload(&dout);

        // 1. dW = matmul_tn(x, dout)
        // x: (batch, in_size), dout: (batch, out_size)
        // TNにより x^T * dout = (in_size, out_size) が直接得られる
        let gdw = gpu.matmul_tn_gpu(&gx, &gdout);
        let gpu_dw = gpu.download(&gdw);

        // 2. dx = matmul(dout, wᵀ)
        // standard matmul_gpu を使うため、メモリ上で連続な W^T を事前作成して転送
        let mut w_t = Array2::<f32>::zeros((out_size, in_size));
        w_t.assign(&w.t());
        let gw_t = gpu.upload(&w_t);
        let gdx = gpu.matmul_gpu(&gdout, &gw_t);
        let gpu_dx = gpu.download(&gdx);

        // 3. db = column_sum(dout)
        let gdb = gpu.column_sum_gpu(&gdout);
        let gpu_db = gpu.download(&gdb);

        // --- 比較 ---
        let diff_dx = cpu_dx
            .iter()
            .zip(gpu_dx.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        let diff_dw = cpu_dw
            .iter()
            .zip(gpu_dw.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        let diff_db = cpu_db
            .iter()
            .zip(gpu_db.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);

        assert!(diff_dx < 1e-3, "dx diff: {diff_dx:e}");
        assert!(diff_dw < 1e-3, "dw diff: {diff_dw:e}");
        assert!(diff_db < 1e-3, "db diff: {diff_db:e}");
    }

    #[test]
    fn test_relu_backward_gpu() {
        let gpu = Gpu::new();
        let (rows, cols) = (100, 50);

        // x には正負を混ぜたランダム値を入れる
        let x: Array2<f32> = Array2::random((rows, cols), StandardNormal);
        let dout: Array2<f32> = Array2::random((rows, cols), StandardNormal);

        // --- CPU ---
        let mut relu = ReluLayer::new();
        // forward を実行して内部にマスク(self.mask)を保存させる
        let _ = Layer::forward(&mut relu, x.clone().into_dyn(), false);
        // backward 実行
        let cpu_dx = Layer::backward(&mut relu, dout.clone().into_dyn())
            .into_dimensionality::<Ix2>()
            .unwrap();

        // --- GPU ---
        // GPU ではマスクの代わりに forward の出力 (act) を用いる
        let mut gx = gpu.upload(&x);
        gpu.relu_gpu(&mut gx); // in-place forward: gx は act になる

        let mut gdout = gpu.upload(&dout);
        gpu.relu_backward_gpu(&mut gdout, &gx);
        let gpu_dx = gpu.download(&gdout);

        assert_eq!(cpu_dx, gpu_dx, "relu backward must be exact match");
    }

    #[test]
    fn test_pool_backward_gpu() {
        let gpu = Gpu::new();

        for (n, c, h, w, ph, pw, stride) in [
            (2, 3, 4, 4, 2, 2, 2), // DeepConvNet実形: 2x2 stride 2
            (2, 2, 7, 7, 2, 2, 2), // 奇数サイズ: 7x7 -> 3x3 に切り捨てられるケース
            (3, 4, 6, 6, 3, 3, 3), // overlap回避のため、forwardのstride=1から3に修正
        ] {
            let pad = 0;
            let x: Array4<f32> = Array4::random((n, c, h, w), StandardNormal);

            // --- CPU ---
            let mut pool = PoolingLayer::new(ph, pw, stride, pad);
            let cpu_out = pool.forward(&x); // 順伝播でargmaxを保存
            let (_, _, oh, ow) = cpu_out.dim();

            let dout: Array4<f32> = Array4::random((n, c, oh, ow), StandardNormal);
            let cpu_dx = pool.backward(&dout);

            // --- GPU ---
            let gx = gpu.upload_image(&x);
            // 順伝播で argmax_buf を生成
            let (_gy, argmax_buf) = gpu.pool_forward_gpu(&gx, ph, pw, stride, pad);

            let gdout = gpu.upload_image(&dout);
            let gdx = gpu.pool_backward_gpu(&gdout, &argmax_buf, (n, c, h, w), ph, pw, stride, pad);
            let gpu_dx = gpu.download(&gdx.tensor);

            // 比較 (N, C*H*W に平坦化)
            let cpu_dx_flat = cpu_dx
                .as_standard_layout()
                .into_owned()
                .into_shape_with_order((n, c * h * w))
                .unwrap();

            assert_eq!(cpu_dx_flat, gpu_dx, "pool backward must be exact match");
        }
    }

    #[test]
    fn test_col2im_gpu() {
        use crate::conv::col2im;

        let gpu = Gpu::new();

        // プローブの4ケース: (n, c, h, w, fh, fw, stride, pad)
        for (n, c, h, w, fh, fw, stride, pad) in [
            (2, 3, 4, 4, 3, 3, 1, 0), // standard
            (2, 3, 4, 4, 3, 3, 1, 1), // pad 処理
            (1, 1, 5, 5, 2, 2, 2, 0), // stride > 1
            (2, 2, 5, 5, 3, 3, 2, 1), // all combined
        ] {
            let oh = (h + 2 * pad - fh) / stride + 1;
            let ow = (w + 2 * pad - fw) / stride + 1;

            // backward では dcol は (n*oh*ow, c*fh*fw) 形状の行列
            let dcol: Array2<f32> = Array2::random((n * oh * ow, c * fh * fw), StandardNormal);

            // --- CPU ---
            let cpu_dx = col2im(&dcol, (n, c, h, w), fh, fw, stride, pad);

            // --- GPU ---
            let gdcol = gpu.upload(&dcol);
            let gdx = gpu.col2im_gpu(&gdcol, (n, c, h, w), fh, fw, stride, pad);
            let gpu_dx = gpu.download(&gdx.tensor);

            // CPU 側の dx を NCHW (N, C*H*W) の平坦なレイアウトに揃える
            let cpu_dx_flat = cpu_dx
                .as_standard_layout()
                .into_owned()
                .into_shape_with_order((n, c * h * w))
                .unwrap();

            assert_eq!(cpu_dx_flat, gpu_dx, "col2im must be exact match");
        }
    }

    #[test]
    fn test_conv_backward_gpu() {
        use crate::conv::ConvolutionLayer;
        use crate::layers::Layer;
        use ndarray::{Axis, Ix4};

        let gpu = Gpu::new();

        let n = 2;
        let c = 3;
        let h = 5;
        let w = 5;
        let fn_ = 4;
        let fh = 3;
        let fw = 3;
        let stride = 1;
        let pad = 1;

        let oh = (h + 2 * pad - fh) / stride + 1;
        let ow = (w + 2 * pad - fw) / stride + 1;

        let x: Array4<f32> = Array4::random((n, c, h, w), StandardNormal);
        let w_arr: Array4<f32> = Array4::random((fn_, c, fh, fw), StandardNormal);
        let b_arr: Array1<f32> = Array1::random(fn_, StandardNormal);
        let dout: Array4<f32> = Array4::random((n, fn_, oh, ow), StandardNormal);

        // --- CPU ---
        let mut conv = ConvolutionLayer::new(w_arr.clone(), b_arr.clone(), stride, pad);
        let _ = Layer::forward(&mut conv, x.clone().into_dyn(), false); // forward で col を計算・保持
        let cpu_dx = Layer::backward(&mut conv, dout.clone().into_dyn())
            .into_dimensionality::<Ix4>()
            .unwrap();

        let cpu_dw = conv.dw(); // (fn_, c, fh, fw)
        let cpu_db = conv.db(); // (fn_,)

        // --- GPU ---
        // 1. 実 forward で col を生成
        let gx = gpu.upload_image(&x);
        let gcol = gpu.im2col_gpu(&gx, fh, fw, stride, pad);

        // 2. w_col を「順伝搬の向き (c*fh*fw, fn_)」でアップロード (dcol 計算用)
        let w_2d = w_arr
            .clone()
            .into_shape_with_order((fn_, c * fh * fw))
            .unwrap();
        let mut w_colt_cpu = ndarray::Array2::<f32>::zeros((c * fh * fw, fn_));
        w_colt_cpu.assign(&w_2d.t());
        let gw_colt = gpu.upload(&w_colt_cpu);

        let gdout = gpu.upload_image(&dout);

        // 3. backward 一括処理
        let (gdx_img, gdw_colt, gdb) =
            gpu.conv_backward_gpu(&gdout, &gcol, &gw_colt, (n, c, h, w), fh, fw, stride, pad);

        let gpu_dx = gpu.download(&gdx_img.tensor);
        let gpu_dw_colt = gpu.download(&gdw_colt);
        let gpu_db = gpu.download(&gdb);

        // --- 比較 ---
        // dx: (n, c*h*w) 比較
        let cpu_dx_flat = cpu_dx
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order((n, c * h * w))
            .unwrap();
        let diff_dx = cpu_dx_flat
            .iter()
            .zip(gpu_dx.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(diff_dx < 1e-3, "dx diff: {diff_dx:e}");

        // dW: CPU の (fn_, c, fh, fw) を (fn_, k) に変形し、転置して (k, fn_) と GPU を比較
        let cpu_dw_2d = cpu_dw
            .clone()
            .into_shape_with_order((fn_, c * fh * fw))
            .unwrap();
        let cpu_dw_colt = cpu_dw_2d.t(); // (c*fh*fw, fn_)
        let diff_dw = cpu_dw_colt
            .iter()
            .zip(gpu_dw_colt.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(diff_dw < 1e-3, "dw diff: {diff_dw:e}");

        // db: CPU の (fn_,) を (1, fn_) に変形して GPU と比較
        let cpu_db_2d = cpu_db.clone().insert_axis(Axis(0));
        let diff_db = cpu_db_2d
            .iter()
            .zip(gpu_db.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        assert!(diff_db < 1e-3, "db diff: {diff_db:e}");
    }

    #[test]
    fn test_sgd_update_gpu() {
        let gpu = Gpu::new();
        let (rows, cols) = (100, 50);
        let lr = 0.01f32;

        let mut param_cpu = Array2::random((rows, cols), StandardNormal);
        let grad = Array2::random((rows, cols), StandardNormal);
        let param_cpu_orig = param_cpu.clone();

        // --- CPU ---
        ndarray::Zip::from(&mut param_cpu)
            .and(&grad)
            .for_each(|p, &g| *p -= lr * g);

        // --- GPU ---
        let mut gparam = gpu.upload(&param_cpu_orig);
        let ggrad = gpu.upload(&grad);

        gpu.sgd_update_gpu(&mut gparam, &ggrad, lr);

        let gpu_param = gpu.download(&gparam);

        // --- 比較 ---
        let max_diff = param_cpu
            .iter()
            .zip(gpu_param.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);

        // FMA (Fused Multiply-Add) 最適化の有無により、CPUとGPUで 1 ULP (1.192e-7) の差異が生じる。
        // 加算順序の変動がなくても完全一致(exact)にはならないため、マシンエプシロン相当の許容誤差を設ける。
        assert!(
            max_diff < 1e-6,
            "SGD update diff is too large: {max_diff:e}"
        );
    }

    #[test]
    fn test_matmul_nt_gpu() {
        let gpu = Gpu::new();
        let m = 100;
        let k = 50;
        let n = 30;

        let a = Array2::random((m, k), StandardNormal);
        let b = Array2::random((n, k), StandardNormal);

        // --- CPU ---
        // A * B^T
        let cpu_c = a.dot(&b.t());

        // --- GPU ---
        let ga = gpu.upload(&a);
        let gb = gpu.upload(&b);
        let gc = gpu.matmul_nt_gpu(&ga, &gb);
        let gpu_c = gpu.download(&gc);

        // --- 比較 ---
        let max_diff = cpu_c
            .iter()
            .zip(gpu_c.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);

        // 蓄積演算（dot/sum）が大量に含まれており、加算順序がCPUの実装と
        // WGSLのナイーブな4並列+スカラー処理とで全く異なるため、完全一致はしない。
        assert!(max_diff < 1e-3, "matmul_nt diff is too large: {max_diff:e}");
    }

    #[test]
    fn test_gpu_adam() {
        use crate::optimizer::{Adam, Optimizer};
        let gpu = Gpu::new();
        
        let mut param_cpu = Array2::random((10, 10), StandardNormal).into_dyn();
        let mut param_gpu = gpu.upload(&param_cpu.clone().into_dimensionality::<ndarray::Ix2>().unwrap());
        
        let mut adam_cpu = Adam::new(0.01);
        let mut adam_gpu = GpuAdam::new();
        
        for i in 0..3 {
            let grad_cpu = Array2::random((10, 10), StandardNormal).into_dyn();
            let grad_gpu = gpu.upload(&grad_cpu.clone().into_dimensionality::<ndarray::Ix2>().unwrap());
            
            adam_cpu.update(&mut param_cpu.view_mut(), &grad_cpu.view());
            adam_gpu.update(&gpu, &mut param_gpu, &grad_gpu, 0.01);
            
            let downloaded = gpu.download(&param_gpu);
            
            let max_diff = param_cpu
                .iter()
                .zip(downloaded.iter())
                .map(|(c, g)| (c - g).abs())
                .fold(0.0f32, f32::max);

            assert!(max_diff < 1e-6, "Adam mismatch at step {}: diff={:e}", i, max_diff);
        }
    }
}
