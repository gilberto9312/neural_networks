# Redes Neuronales en Rust: Un Reto de 21 Días

Este repositorio documenta un reto de 21 días para aprender los conceptos fundamentales de las redes neuronales desde cero, utilizando Rust. Puedes seguir el reto día a día o consultar la guía temática para encontrar conceptos específicos.

## 🚀 Cómo Empezar

Cada proyecto es una aplicación de Rust independiente. Para ejecutar cualquiera de ellos, sigue estos pasos:

1.  Navega al directorio del proyecto que te interese.
2.  Ejecuta el proyecto usando Cargo.

```bash
# Ejemplo para ejecutar el proyecto del Día 5: Perceptrón
cd perceptron
cargo run
```

---

## 🗓️ Cronología del Reto (Día a Día)

Esta es la progresión recomendada para seguir el reto de forma secuencial.

| Día | Proyecto | Concepto Clave |
| :-- | :--- | :--- |
| 1 | [`neurons`](./neurons/) | Neurona Artificial |
| 2 | [`sigmoid`](./sigmoid/) | Función de Activación Sigmoid |
| 3 | [`tanh`](./tanh/) | Función de Activación Tanh |
| 4 | [`relu`](./relu/) | Función de Activación ReLU |
| 5 | [`perceptron`](./perceptron/) | Perceptrón Simple y Aprendizaje |
| 6 | [`xor`](./xor/) | Red Neuronal Multicapa |
| 7 | [`momentum`](./momentum/) | Optimizador Momentum |
| 8 | [`comparation-optimizer`](./comparation-optimizer/) | Comparación de Optimizadores (Momentum, RMSprop, Adam) |
| 9 | [`regularization`](./regularization/) | Regularización (L1, L2, Dropout) |
| 10 | [`graph`](./graph/) | Visualización de Métricas de Entrenamiento |
| 11 | [`earlyStopping`](./earlyStopping/) | Parada Temprana (Early Stopping) |
| 12 | [`dataset_iris`](./dataset_iris/) | Aplicación Práctica: Clasificación Multiclase |
| 13 | [`dataset_mnist`](./dataset_mnist/) | Aplicación Práctica: Clasificación png  |

---

## 📚 Guía Temática de Conceptos

Usa este índice para encontrar proyectos relacionados con un tema específico.

### 1. Fundamentos de Redes Neuronales
*   **[`neurons`](./neurons/)**: La unidad básica de una red neuronal.
*   **[`perceptron`](./perceptron/)**: Un modelo de una sola neurona que puede aprender problemas linealmente separables.
*   **[`xor`](./xor/)**: La necesidad de capas ocultas para resolver problemas no lineales.

### 2. Funciones de Activación
*   **[`sigmoid`](./sigmoid/)**: Transforma salidas a un rango de probabilidad (0, 1).
*   **[`tanh`](./tanh/)**: Una alternativa a Sigmoid centrada en cero (-1, 1).
*   **[`relu`](./relu/)**: La función de activación más popular por su eficiencia y simplicidad.

### 3. Optimización del Entrenamiento
*   **[`momentum`](./momentum/)**: Acelera el entrenamiento añadiendo "inercia" a la actualización de pesos.
*   **[`comparation-optimizer`](./comparation-optimizer/)**: Compara el rendimiento de Momentum, RMSprop y Adam.

### 4. Técnicas de Regularización (Prevención de Overfitting)
*   **[`regularization`](./regularization/)**: Implementa L1, L2 y Dropout para mejorar la generalización del modelo.
*   **[`earlyStopping`](./earlyStopping/)**: Detiene el entrenamiento de forma inteligente para evitar el sobreajuste.

### 5. Análisis y Aplicaciones Prácticas
*   **[`graph`](./graph/)**: Visualiza las curvas de error para diagnosticar el entrenamiento.
*   **[`dataset_iris`](./dataset_iris/)**: Resuelve un problema de clasificación del mundo real de principio a fin.
*   **[`dataset_mnist`](./dataset_mnist/)**: Resuelve un problema de clasificación de imagenes.