# Dataset Mnist

Este proyecto implementa una red neuronal en Rust para clasificar imágenes de dígitos escritos a mano del dataset MNIST. El código está autocontenido y no depende de frameworks externos de machine learning.

## Explicación del Código

El archivo `src/main.rs` contiene la implementación completa de la red neuronal y el ciclo de entrenamiento. Los componentes principales son:

-   **`NeuralNetwork`**: Estructura principal que define la red, incluyendo sus capas y el optimizador.
-   **`Layer`**: Define una capa de neuronas, con sus pesos, sesgos y funciones de activación (ReLU para capas ocultas y Softmax para la capa de salida).
-   **`AdamOptimizer`**: Implementa el optimizador Adam para ajustar eficientemente los pesos de la red durante el entrenamiento.
-   **`train` y `validate`**: Funciones que manejan el ciclo de entrenamiento y la evaluación del modelo contra un conjunto de datos de validación.
-   **`load_mnist_from_png_improved`**: Función encargada de cargar las imágenes PNG, procesarlas, normalizar los valores de los píxeles y dividirlas en conjuntos de entrenamiento y validación.
-   **`plot_training_metrics`**: Utiliza la librería `plotters` para generar gráficos que muestran la pérdida (loss) y la precisión (accuracy) a lo largo de las épocas de entrenamiento.

## ¿Qué aprendimos aquí?

A través de este ejemplo, podemos aprender varios conceptos fundamentales del deep learning:

1.  **Arquitectura de una Red Neuronal**: Cómo estructurar capas de neuronas para procesar datos complejos como imágenes.
2.  **Forward y Backward Propagation**: El proceso de pasar datos a través de la red para hacer una predicción (forward) y luego calcular los errores para ajustar los pesos (backward).
3.  **Funciones de Activación**: La importancia de funciones no lineales como ReLU y cómo Softmax es útil para problemas de clasificación multiclase.
4.  **Optimización**: Cómo algoritmos como Adam ayudan a la red a converger hacia una solución óptima de manera más rápida y estable que un descenso de gradiente estándar.
5.  **Ciclo de Vida de un Modelo**: El proceso completo de cargar datos, entrenar un modelo, validarlo para evitar el sobreajuste (overfitting) y visualizar su rendimiento.
6.  **Procesamiento de Datos**: La importancia de normalizar los datos de entrada para mejorar el rendimiento y la estabilidad del entrenamiento.

## Estructura de la Carpeta del Proyecto

```
dataset_mnist/
├── Cargo.lock
├── Cargo.toml
├── content.json
├── mnist_small_dataset_metrics.png
├── mnist_training_metrics.png
├── README.md
├── src/
│   └── main.rs
└── public/
    └── train/
       ├── 0/
       ├── 1/
       └── ... (más carpetas por dígito)
```

## Configuración del Dataset

**Importante**: El dataset de imágenes MNIST no se incluye en este repositorio debido a su tamaño. Sin embargo, puedes descargarlo y configurarlo fácilmente.

1.  **Descarga el dataset** desde el siguiente enlace de Kaggle, que contiene las imágenes en formato `.zip`:
    *   [https://www.kaggle.com/datasets/playlist/mnistzip](https://www.kaggle.com/datasets/playlist/mnistzip)

2.  **Estructura de carpetas**: Descomprime los archivos y organízalos dentro de la carpeta `public/` siguiendo la estructura mostrada arriba. Debes tener una carpeta `train` y una `valid` (o solo `train` si prefieres que el código haga la división automáticamente), y dentro de ellas, subcarpetas nombradas del `0` al `9` que contengan las imágenes correspondientes a cada dígito.

Al finalizar el entrenamiento, el programa generará una imagen llamada `mnist_training_metrics.png` con los gráficos de rendimiento del modelo, similar a esta:

### Configuración del Test

🔄 Procesando: 9/9998.png (28x28) 

📸 Procesando: public/train/9/9998.png

    📊 Estadísticas de 9998.png: Media=0.404, Std=0.266, Min=-0.500, Max=0.500, Píxeles activos=739/784
    ✅ Cargada como dígito 9

🎉 === Datos Cargados Exitosamente ===

📈 Total de muestras: 60000

📊 Análisis Estadístico del Dataset:

  📋 Distribución por clase:

    Dígito 0: 5923 muestras
    Dígito 1: 6742 muestras
    Dígito 2: 5958 muestras
    Dígito 3: 6131 muestras
    Dígito 4: 5842 muestras
    Dígito 5: 5421 muestras
    Dígito 6: 5918 muestras
    Dígito 7: 6265 muestras
    Dígito 8: 5851 muestras
    Dígito 9: 5949 muestras

  📈 Estadísticas globales de píxeles:

    Media: 0.3693
    Desviación estándar: 0.3081
    Rango: [-0.5000, 0.5000]

📊 División Final:

  🏋️  Entrenamiento: 48000 muestras (80.0%)

  🧪 Validación: 12000 muestras (20.0%)

⚙️  Configuración para dataset de 60000 muestras:

  🏗️  Arquitectura: 784 -> 128 -> 64 -> 10
  📚 Tasa de aprendizaje: 0.001
  🔄 Épocas: 500
  📦 Tamaño de batch: 4

🧠 Creando red neuronal...


