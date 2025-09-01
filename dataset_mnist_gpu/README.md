# Dataset MNIST con Aceleración GPU

Este proyecto es una adaptación de la red neuronal anterior para clasificar dígitos del dataset MNIST, pero modificada para aprovechar la potencia de cálculo de la GPU. El objetivo principal es acelerar la operación más costosa del entrenamiento: la multiplicación de matrices.

## ¿Qué aprendimos aquí?

Además de los conceptos del proyecto anterior, esta implementación introduce varias ideas clave sobre computación en GPU:

1.  **Computación en Paralelo**: Entender cómo las GPUs, con sus miles de núcleos, pueden realizar operaciones matemáticas (como la multiplicación de matrices) de forma masivamente paralela, resultando en una aceleración significativa.
2.  **Abstracción de GPU con `wgpu`**: Se utiliza la librería `wgpu` de Rust, que actúa como una capa de abstracción sobre las APIs nativas de la GPU (Metal en macOS, Vulkan en Linux/Windows, DirectX 12 en Windows). Esto permite escribir código portable que se ejecuta en diferente hardware.
3.  **Gestión de Memoria CPU-GPU**: Los datos (pesos de la red, entradas) deben ser transferidos explícitamente desde la memoria RAM del sistema a la VRAM de la GPU a través de "buffers". Después del cálculo, los resultados deben ser copiados de vuelta.
4.  **Compute Shaders (WGSL)**: La lógica que se ejecuta en la GPU se escribe en un lenguaje específico llamado WGSL (WebGPU Shading Language). En `matmul.wgsl`, se define un pequeño programa que cada hilo de la GPU ejecuta para calcular un único valor de la matriz de resultado.
5.  **Pipeline de Cómputo**: Se aprende a configurar un "pipeline" que le dice a la GPU cómo ejecutar nuestro shader, incluyendo cómo están organizados los datos de entrada y salida (a través de `BindGroup`).

## Diferencias Clave con la Versión CPU

-   **Backend de Cómputo**: La versión CPU realiza todos los cálculos en el procesador principal. Esta versión descarga la multiplicación de matrices (la parte más pesada del `forward pass`) a la GPU.
-   **Flujo de Datos**: Se añade un `GpuBackend` (`gpu_backend.rs`) que gestiona la comunicación con la GPU. La función `forward_gpu` se encarga de enviar los datos a la GPU, ejecutar el shader y recibir el resultado.
-   **Operación Acelerada**: Solo la multiplicación de matrices se ha movido a la GPU. La retropropagación (`backward pass`) y la actualización de pesos con el optimizador Adam todavía se ejecutan en la CPU. Esto es una optimización común, ya que el `forward pass` con grandes matrices es el principal cuello de botella.

## ¿Por Qué Usar la GPU?

El entrenamiento de redes neuronales implica una cantidad masiva de operaciones de álgebra lineal, especialmente multiplicaciones de matrices. Mientras que una CPU tiene unos pocos núcleos potentes diseñados para tareas secuenciales, una GPU tiene miles de núcleos más simples optimizados para ejecutar la misma operación sobre muchos datos a la vez. Al mover la multiplicación de matrices a la GPU, podemos pasar de un cálculo secuencial a uno masivamente paralelo, reduciendo drásticamente el tiempo de entrenamiento.

## Implementación Específica y Posibles Mejoras

Este código fue desarrollado y probado en un **Mac con chip M1**. La librería `wgpu` detecta automáticamente el sistema operativo y utiliza la API **Metal** de Apple como backend.

**Mejora para otras GPUs (NVIDIA, AMD):**

La gran ventaja de usar `wgpu` es que el código **ya es mayormente compatible** con otras plataformas. No se necesita una reescritura fundamental. Para ejecutarlo en un sistema con una GPU NVIDIA o AMD (bajo Windows o Linux), `wgpu` utilizaría automáticamente el backend **Vulkan** o **DirectX 12**.

La "mejora" no consistiría en cambiar el código Rust o el shader WGSL, sino en:
1.  **Compilar y Probar**: Asegurarse de que el proyecto compila y se ejecuta correctamente en el sistema operativo de destino (Windows/Linux).
2.  **Instalación de Drivers**: Verificar que los drivers de la GPU y las librerías de Vulkan estén correctamente instalados y actualizados.

El shader `matmul.wgsl` es portable por diseño y funcionará sin cambios en cualquier GPU compatible con WebGPU.

## Cómo Ejecutar

Asegúrate de tener el dataset configurado como se describe en el `README.md` del proyecto `dataset_mnist`. Luego, ejecuta el proyecto con:

```bash
cargo run 
```

El programa detectará y utilizará la GPU disponible para acelerar el entrenamiento.

## Resultado esperado,
![uso del gpu](https://github.com/gilberto9312/neural_networks/blob/main/dataset_mnist_gpu/public/Screenshot%202025-08-31%20at%207.25.18%E2%80%AFPM.png)


