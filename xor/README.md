# Red Neuronal Multicapa en Rust 🧠

Una implementación desde cero de una red neuronal multicapa en Rust, sin librerías externas, diseñada para aprender la función XOR mediante backpropagation.

## 📋 Descripción del Proyecto

Este proyecto implementa una red neuronal feedforward de 3 capas que puede resolver problemas no linealmente separables como XOR. La implementación incluye:

- ✅ Forward propagation
- ✅ Backpropagation con gradient descent
- ✅ Función de activación sigmoide
- ✅ Entrenamiento completo con múltiples épocas
- ✅ Sin dependencias externas (solo `std`)

## 🏗️ Arquitectura de la Red

```
Entrada (2) → Capa Oculta (3) → Salida (1)
```

### ¿Por qué **2 entradas**?

Porque la red está resolviendo la **función lógica XOR**, que recibe **dos valores binarios** de entrada:

| Entrada A | Entrada B | XOR |
|-----------|-----------|-----|
| 0         | 0         | 0   |
| 0         | 1         | 1   |
| 1         | 0         | 1   |
| 1         | 1         | 0   |

→ Entonces tu input es un **vector de 2 valores** `[A, B]`  
Ejemplo: `[1.0, 0.0]` representa `1 XOR 0`.

### ✅ ¿Por qué **1 salida**?

Porque quieres que la red te diga el **resultado del XOR**:
* Si es `1`, la salida debe acercarse a `1.0`
* Si es `0`, la salida debe acercarse a `0.0`

Entonces, **una única neurona de salida** basta.

### ✅ ¿Por qué **3 neuronas ocultas**?

La función XOR **no es linealmente separable**, así que una red sin capa oculta no puede aprenderla.

* Con **1 o 2 neuronas ocultas**, a veces no es suficiente.
* Con **3 neuronas**, ya tienes el poder necesario para modelar la "curva" de XOR.

Con 3 se resuelve bien, con 4 aprendes más rápido, con más de 5 para XOR es innecesario.

## ⚙️ Parámetros Clave

### 🧠 ¿Qué es el `learning_rate`?

El `learning_rate` o **tasa de aprendizaje** es **cuánto cambia la red cada vez que aprende** de un error.

* Si es muy **pequeño** (`0.0001`), aprende lentamente.
* Si es muy **grande** (`10.0`), puede volverse inestable y no aprender nada.

Un valor típico en redes pequeñas como la tuya está entre `0.01` y `1.5`.  
En tu caso, usas `1.0`, que es bastante alto pero puede funcionar porque el problema es pequeño.

### 🔢 Función de Costo: Mean Squared Error (MSE)

```rust
total_error += 0.5 * error * error;
```

**¿Por qué `0.5 * error²`?**
- El `0.5` hace que la derivada sea más limpia: cuando derivas, el 2 del exponente cancela el 0.5


### Salida Esperada

```
=== Red Neuronal Multicapa para XOR ===

Datos de entrenamiento (función XOR):
  [0.0, 0.0] → [0.0]
  [0.0, 1.0] → [1.0]
  [1.0, 0.0] → [1.0]
  [1.0, 1.0] → [0.0]

Época 0: Error total = 1.234567
Época 500: Error total = 0.123456
...
Época 4500: Error total = 0.000123

=== Resultados después del entrenamiento ===

Pruebas finales:
Entrada: [0.0, 0.0] → Salida: 0.0123, Predicción: 0, Esperado: 0, ✓
Entrada: [0.0, 1.0] → Salida: 0.9876, Predicción: 1, Esperado: 1, ✓
Entrada: [1.0, 0.0] → Salida: 0.9845, Predicción: 1, Esperado: 1, ✓
Entrada: [1.0, 1.0] → Salida: 0.0234, Predicción: 0, Esperado: 0, ✓

🎉 Resultado: ¡La red aprendió XOR correctamente!
```

### Ajustar Learning Rate
```rust
let mut network = NeuralNetwork::new(2, 3, 1, 0.1);  // Más conservador
let mut network = NeuralNetwork::new(2, 3, 1, 2.0);  // Más agresivo
```


## ⚠️ Escalabilidad: ¿Qué pasa con muchas capas?

Si quisieras **muchas capas ocultas (deep learning)**, aparecen nuevos desafíos:

### Problemas con Redes Profundas
* **Vanishing Gradient**: El gradiente se vuelve muy pequeño en capas tempranas
* **Exploding Gradient**: El gradiente crece exponencialmente
* **Convergencia lenta**: Más parámetros = más tiempo de entrenamiento



**Para XOR, 99 capas es como usar un tanque para abrir una lata.**

## 📚 Conceptos Implementados

### Forward Propagation
1. **Entrada → Capa Oculta**: `hidden[i] = sigmoid(sum(input[j] * weight[j][i]) + bias[i])`
2. **Capa Oculta → Salida**: `output[i] = sigmoid(sum(hidden[j] * weight[j][i]) + bias[i])`

### Backpropagation
1. **Error en salida**: `(target - output) * output * (1 - output)`
2. **Error en capa oculta**: `suma_errores_salida * peso * hidden * (1 - hidden)`
3. **Actualizar pesos**: `peso_nuevo = peso_viejo + learning_rate * error * activación`

### Función Sigmoide y su Derivada
- **Sigmoide**: `σ(x) = 1/(1 + e^(-x))`
- **Derivada**: `σ'(x) = σ(x) * (1 - σ(x))`
*   [Día 2: Sigmoid](../sigmoid/README.md)

## 🎯 Próximos Pasos

1. **Diferentes funciones de activación**: ReLU, tanh, Leaky ReLU
2. **Optimizadores avanzados**: Momentum, Adam, RMSprop
3. **Regularización**: L1, L2, Dropout
4. **Redes más profundas**: Múltiples capas ocultas
5. **Datasets reales**: MNIST, clasificación de imágenes

## 📈 Métricas de Rendimiento

- **Convergencia típica**: 2000-5000 épocas para XOR
- **Precisión esperada**: >99% en datos de entrenamiento
- **Tiempo de ejecución**: <1 segundo en hardware moderno

## 🐛 Debugging

Si la red no converge:
1. **Reduce el learning rate** (prueba 0.1 o 0.01)
2. **Aumenta las épocas** (10,000 en vez de 5,000)
3. **Cambia la inicialización** de pesos
4. **Añade más neuronas** ocultas (4 o 5)


---

*Implementación creada como parte de un reto de 21 días aprendiendo redes neuronales desde cero.*