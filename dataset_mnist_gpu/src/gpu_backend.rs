//! Módulo para la gestión de cálculos en la GPU utilizando wgpu.
//!
//! Este backend abstrae la inicialización de la GPU y la ejecución de
//! tareas de cómputo, como la multiplicación de matrices, permitiendo que
//! la red neuronal descargue el trabajo pesado del CPU a la GPU.

use wgpu::util::DeviceExt;
use bytemuck;

/// Estructura que contiene los componentes esenciales de wgpu para interactuar con la GPU.
pub struct GpuBackend {
    /// El dispositivo lógico, nuestra conexión principal con la GPU.
    pub device: wgpu::Device,
    /// La cola de comandos, donde se envían las tareas a la GPU para su ejecución.
    pub queue: wgpu::Queue,
}

impl GpuBackend {
    /// Inicializa el backend de la GPU de forma asíncrona.
    ///
    /// Busca un adaptador de GPU disponible (el hardware físico), solicita un
    /// dispositivo lógico para comunicarse con él y establece una cola de comandos.
    pub async fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        
        // Un adaptador es un manejador para el hardware de la GPU física.
        // Solicitamos uno con preferencia de "Alto Rendimiento".
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;
        
        // Imprime el nombre de la GPU que se está utilizando para confirmación.
        println!("✅ Using GPU: {}", adapter.get_info().name);

        // Un dispositivo es la conexión lógica con la GPU, que nos permite crear recursos.
        // La cola es donde enviamos los comandos (ej. "ejecuta este shader").
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .ok()?;
        Some(Self { device, queue })
    }

    /// Ejecuta la multiplicación de matrices (A * B) en la GPU.
    ///
    /// Esta es la operación clave de optimización. En lugar de hacer miles de
    /// multiplicaciones y sumas en el CPU, se envían las matrices A y B a la GPU,
    /// que puede realizar estas operaciones en paralelo masivamente, resultando
    /// en una aceleraci��n significativa.
    ///
    /// # Arguments
    /// * `a` - Matriz A como un slice de f32.
    /// * `b` - Matriz B (en este caso, un vector) como un slice de f32.
    /// * `m`, `n`, `k` - Dimensiones de las matrices (A: m x k, B: k x n).
    pub async fn matmul(&self, a: &[f32], b: &[f32], m: u32, n: u32, k: u32) -> Vec<f32> {
        let c_size = (m * n) as usize;

        // --- 1. Transferencia de Datos: CPU -> GPU ---
        // Creamos buffers en la memoria de la GPU y copiamos los datos de las matrices
        // desde la memoria RAM del CPU a la VRAM de la GPU.
        let a_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix A Buffer"),
            contents: bytemuck::cast_slice(a),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let b_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix B Buffer"),
            contents: bytemuck::cast_slice(b),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Buffer para el resultado (matriz C), solo se asigna en la GPU.
        let c_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix C Buffer"),
            size: (c_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Buffer para las dimensiones, usado por el shader para conocer los límites.
        let dims = [m, n, k];
        let dims_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dimensions Buffer"),
            contents: bytemuck::cast_slice(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // --- 2. Preparación del Cómputo en GPU ---
        // Cargamos el código del shader WGSL que contiene la lógica de la multiplicación.
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("matmul.wgsl").into()),
        });

        // Creamos un "pipeline" de cómputo, que es el estado configurable de la GPU
        // para ejecutar nuestro shader.
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        // Un "bind group" conecta los recursos (buffers) que creamos con las variables
        // declaradas dentro del shader. Es el "cableado" entre los datos y el código.
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: c_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dims_buf.as_entire_binding() },
            ],
            label: Some("MatMul BindGroup"),
        });

        // --- 3. Ejecución del Comando en la GPU ---
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MatMul Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MatMul Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            // ¡Aquí es donde se lanza el trabajo!
            // Le decimos a la GPU que ejecute nuestro shader en una cuadrícula de hilos.
            // Cada hilo calculará un elemento de la matriz de resultado.
            compute_pass.dispatch_workgroups((m + 7) / 8, (n + 7) / 8, 1);
        }

        // --- 4. Transferencia de Datos: GPU -> CPU ---
        // Creamos un buffer en el CPU para recibir el resultado.
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (c_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Copiamos el resultado del buffer de la GPU al buffer del CPU.
        encoder.copy_buffer_to_buffer(&c_buf, 0, &output_buf, 0, (c_size * 4) as u64);
        
        // Enviamos todos los comandos grabados a la cola de la GPU para su ejecución.
        self.queue.submit(Some(encoder.finish()));

        // Esperamos a que la GPU termine y leemos los datos de vuelta.
        let buffer_slice = output_buf.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| ());
        self.device.poll(wgpu::Maintain::Wait);

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        output_buf.unmap();

        result
    }
}
