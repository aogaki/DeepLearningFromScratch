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

pub struct Gpu {
    pub device: Device,
    pub queue: Queue,
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

        Gpu { device, queue }
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
