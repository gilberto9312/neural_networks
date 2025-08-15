# 🌸 Red Neuronal para Clasificación del Dataset Iris 🌸

Una implementación de una red neuronal en Rust desde cero, diseñada para resolver el clásico problema de clasificación multiclase del dataset Iris.

## 🎯 Descripción

Este proyecto construye, entrena y evalúa una red neuronal feedforward para clasificar las flores de Iris en tres especies diferentes (Setosa, Versicolor y Virginica) basándose en 3 características físicas: longitud, ancho de sépalos y pétalos.

El programa realiza las siguientes tareas:
1.  **Carga y Preprocesa los Datos**: Lee el archivo `dataset.csv`, normaliza las características usando escalado min-max y divide los datos en conjuntos de entrenamiento y prueba.
2.  **Entrena el Modelo**: Utiliza backpropagation y el optimizador Adam para ajustar los pesos de la red.
3.  **Genera una Gráfica**: Crea un archivo `training_curve.png` que muestra la evolución del error (pérdida) durante el entrenamiento.
4.  **Evalúa el Rendimiento**: Mide la precisión del modelo entrenado en el conjunto de prueba.

## 🏗️ Arquitectura de la Red

-   **Capa de entrada**: 4 neuronas (corresponden a las 4 características del dataset).
-   **Capas ocultas**: 2 capas ocultas con 16 y 8 neuronas respectivamente.
-   **Función de activación (oculta)**: **ReLU (Rectified Linear Unit)**.
-   **Capa de salida**: 3 neuronas (una para cada especie de Iris).
-   **Función de activación (salida)**: **Softmax**, para obtener una distribución de probabilidad sobre las clases.

## 🧠 Fórmulas Clave

### 1. Función de Activación ReLU

Para las capas ocultas, se utiliza la función ReLU, que introduce no linealidad y es computacionalmente eficiente.

```
ReLU(x) = max(0, x)
```

Su derivada es simple, lo que acelera el proceso de backpropagation:

```
ReLU'(x) = 1 si x > 0, 0 si x ≤ 0
```

### 2. Función de Activación Softmax

En la capa de salida, Softmax convierte las salidas brutas (logits) en probabilidades, asegurando que la suma de todas las probabilidades sea 1.

```
Softmax(z_i) = e^(z_i) / Σ(e^(z_j))
```

### 3. Función de Pérdida (Cross-Entropy)

Para medir el error en problemas de clasificación multiclase, se utiliza la pérdida de entropía cruzada. Compara la distribución de probabilidad predicha con la distribución real (one-hot encoded).

```
L = -Σ(y_true * log(y_pred))
```

### 4. Optimizador Adam

Adam (Adaptive Moment Estimation) es un algoritmo de optimización eficiente que ajusta la tasa de aprendizaje para cada peso de la red. Combina las ventajas de dos extensiones del descenso de gradiente estocástico: Momentum y RMSprop.

```
m = β₁·m + (1-β₁)·g          // Estimación del primer momento (media)
v = β₂·v + (1-β₂)·g²         // Estimación del segundo momento (varianza)
m_hat = m / (1 - β₁^t)       // Corrección de sesgo
v_hat = v / (1 - β₂^t)
w = w - η · m_hat / (√v_hat + ε)
```

## 📊 Salida del Programa

```
🌸 Red Neuronal para Dataset Iris 🌸
Cargando dataset...
Dataset cargado exitosamente!
Entrenamiento: 120 muestras
Prueba: 30 muestras
Entrenamiento: 105 muestras
Validación: 15 muestras
Iniciando entrenamiento...
Epoch 0: Train Loss = 1.1119, Val Loss = 1.1193, Accuracy = 32.38%
Epoch 100: Train Loss = 0.3691, Val Loss = 0.3599, Accuracy = 97.14%
Epoch 200: Train Loss = 0.1847, Val Loss = 0.1699, Accuracy = 98.10%
...
Epoch 9900: Train Loss = 0.0139, Val Loss = 0.0063, Accuracy = 100.00%

📊 Generando gráfica de entrenamiento...

🎯 Evaluando en conjunto de prueba...
Precisión en prueba: 100.00%
Precisión en entrenamiento: 99.17%

🔍 Ejemplo de predicción:
Muestra: [0.2222, 0.6250, 0.0677, 0.0416]
Clase real: 0 (Setosa)
Clase predicha: 0 (Setosa)
Probabilidades: [0.99, 0.01, 0.00]
```


## 🔧 Experimentación y Variabilidad de Resultados

Los resultados de una red neuronal son muy sensibles a su configuración. Puedes obtener diferentes resultados si modificas los siguientes hiperparámetros en `main.rs`:

### 1. Número de Neuronas
Cambiar el tamaño de las capas ocultas en la función `IrisNeuralNetwork::new()` puede afectar la capacidad del modelo.
-   **Menos neuronas**: Puede llevar a *underfitting* (el modelo es demasiado simple para capturar la complejidad de los datos).
-   **Más neuronas**: Puede llevar a *overfitting* (el modelo memoriza los datos de entrenamiento y no generaliza bien a datos nuevos), además de aumentar el tiempo de entrenamiento.

```rust
// En IrisNeuralNetwork::new()
let layer_sizes = vec![4, 16, 8, 3]; // Prueba cambiando 16 y 8
```

### 2. Tasa de Aprendizaje (Learning Rate)
La tasa de aprendizaje (`learning_rate`) controla cuán grandes son los ajustes a los pesos en cada iteración.
-   **Tasa muy alta**: El modelo puede converger rápidamente pero corre el riesgo de "pasarse" del mínimo óptimo, provocando inestabilidad.
-   **Tasa muy baja**: El entrenamiento será más lento y puede quedarse atascado en mínimos locales.

```rust
// En main()
let mut nn = IrisNeuralNetwork::new(0.001); // Prueba con 0.01, 0.0005, etc.
```

### 3. Función de Activación (ReLU vs. Sigmoid)
Actualmente, el modelo usa **ReLU**. Si la cambias por **Sigmoid**, el comportamiento del entrenamiento cambiará.
-   **Sigmoid**: Comprime los valores a un rango entre 0 y 1. Es útil pero puede sufrir del "problema de desvanecimiento del gradiente" (vanishing gradient), donde los gradientes se vuelven muy pequeños, ralentizando o deteniendo el aprendizaje, especialmente en redes profundas.
-   **ReLU**: Es menos susceptible a este problema y a menudo lleva a una convergencia más rápida.

Para cambiar a Sigmoid, deberás modificar la llamada a la función de activación en el método `forward()` y la de su derivada en `backward()`.

```rust
// En IrisNeuralNetwork::forward()
// Cambia esto:
layer_output = layer_output.iter().map(|&x| relu(x)).collect();
// Por esto:
layer_output = layer_output.iter().map(|&x| sigmoid(x)).collect();

// Y en IrisNeuralNetwork::backward()
// Cambia esto:
next_delta[j] *= relu_derivative(layer_input[j]);
// Por esto:
next_delta[j] *= sigmoid_derivative(layer_input[j]);
```
