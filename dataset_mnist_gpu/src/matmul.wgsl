// Este es un Compute Shader escrito en WGSL (WebGPU Shading Language).
// Su propósito es realizar la multiplicación de matrices (C = A * B) de manera
// masivamente paralela en la GPU.

// --- Declaración de Buffers ---
// Estos 'bindings' conectan las variables del shader con los buffers de memoria
// que se le pasan desde el código Rust a través del BindGroup.

// @group(0) @binding(0): Matriz de entrada A (solo lectura).
@group(0) @binding(0) var<storage, read> a: array<f32>;
// @group(0) @binding(1): Matriz de entrada B (solo lectura).
@group(0) @binding(1) var<storage, read> b: array<f32>;
// @group(0) @binding(2): Matriz de resultado C (lectura y escritura).
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

// @group(0) @binding(3): Un buffer 'uniform' que contiene las dimensiones
// de las matrices para que el shader sepa cómo indexar los arrays.
@group(0) @binding(3) var<uniform> dims: vec3<u32>; 
// dims.x = M (filas de A y C)
// dims.y = N (columnas de B y C)
// dims.z = K (columnas de A y filas de B)

// --- Punto de Entrada del Shader ---
// @compute: Indica que es un compute shader.
// @workgroup_size(8, 8): Define el tamaño del "workgroup" o grupo de trabajo.
// La GPU ejecuta muchos de estos grupos en paralelo. Cada grupo contiene 8x8 = 64 hilos.
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // `gid` es el identificador único global para este hilo específico en la cuadrícula de ejecución.
    // Lo usamos para determinar qué celda de la matriz de resultado (C) debe calcular este hilo.
    let row = gid.x; // Fila de la matriz C
    let col = gid.y; // Columna de la matriz C

    // Guarda de seguridad: nos aseguramos de que el hilo no intente escribir fuera
    // de los límites de la matriz de resultado.
    if (row < dims.x && col < dims.y) {
        var sum: f32 = 0.0;
        // Este bucle calcula el producto punto de la fila `row` de A y la columna `col` de B.
        // Esta es la operación fundamental de la multiplicación de matrices.
        for (var k: u32 = 0u; k < dims.z; k = k + 1u) {
            sum = sum + a[row * dims.z + k] * b[k * dims.y + col];
        }
        // El resultado del producto punto se almacena en la celda correspondiente de la matriz C.
        c[row * dims.y + col] = sum;
    }
}
