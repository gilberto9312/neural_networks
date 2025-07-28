# Neural Network Optimizers Comparison 🧠

Una implementación desde cero en Rust que compara tres algoritmos de optimización populares para redes neuronales: **Momentum**, **RMSprop** y **Adam**.

## 🎯 Descripción

Este proyecto implementa una red neuronal feedforward simple y entrena tres copias idénticas usando diferentes optimizadores en hilos paralelos. El objetivo es comparar su rendimiento al resolver el problema clásico de la función XOR.

### ¿Por qué XOR?

La función XOR es un problema no linealmente separable que requiere al menos una capa oculta para ser resuelto. Es un benchmark clásico para evaluar algoritmos de aprendizaje de redes neuronales.

| Entrada A | Entrada B | Salida |
|-----------|-----------|--------|
| 0         | 0         | 0      |
| 0         | 1         | 1      |
| 1         | 0         | 1      |
| 1         | 1         | 0      |

## 🏗️ Arquitectura de la Red

- **Capa de entrada**: 2 neuronas
- **Capa oculta**: 4 neuronas (función de activación: sigmoid)
- **Capa de salida**: 1 neurona (función de activación: sigmoid)
- **Inicialización**: Pesos aleatorios con semilla fija para reproducibilidad

## 🚀 Algoritmos de Optimización

### 1. Momentum Optimizer
```
velocidad = β × velocidad_anterior + gradiente_actual
peso = peso - learning_rate × velocidad
```
- **β**: 0.9 (factor de momentum)
- **Ventaja**: Suaviza oscilaciones, ayuda a escapar mínimos locales
- **Desventaja**: Puede oscilar en valles estrechos

### 2. RMSprop Optimizer
```
cache = α × cache_anterior + (1-α) × gradiente²
peso = peso - learning_rate × gradiente / √(cache + ε)
```
- **α**: 0.9 (factor de decaimiento)
- **ε**: 1e-8 (para estabilidad numérica)
- **Ventaja**: Se adapta automáticamente a diferentes escalas de gradientes
- **Desventaja**: Puede convergir prematuramente en algunos casos

### 3. Adam Optimizer (Adaptive Moment Estimation)
```
m = β₁ × m_anterior + (1-β₁) × gradiente          // momentum
v = β₂ × v_anterior + (1-β₂) × gradiente²         // varianza
m_corregido = m / (1 - β₁^t)                      // corrección de sesgo
v_corregido = v / (1 - β₂^t)
peso = peso - learning_rate × m_corregido / (√v_corregido + ε)
```
- **β₁**: 0.9 (factor de momentum)
- **β₂**: 0.999 (factor de decaimiento para la varianza)
- **ε**: 1e-8 (para estabilidad numérica)
- **Ventaja**: Combina lo mejor de Momentum y RMSprop
- **Desventaja**: Más complejo computacionalmente

## 🛠️ Características Técnicas

### Concurrencia
- Entrenamiento en **3 hilos paralelos** simultáneos
- Comunicación entre hilos usando `std::sync::mpsc` channels
- Sincronización de resultados con `Arc<Mutex<Vec<T>>>`

### Implementación
- **Generador de números aleatorios personalizado**: Linear Congruential Generator (LCG)
- **Sin dependencias externas**: Implementación completamente desde cero
- **Trait-based polymorphism**: Interfaz común para todos los optimizadores
- **Memory safety**: Aprovecha las garantías de seguridad de memoria de Rust

## 📊 Salida del Programa

```
🧠 COMPARACIÓN DE OPTIMIZADORES: Momentum vs RMSprop vs Adam
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Datos de entrenamiento (función XOR):
   [0.0, 0.0] → [0.0]
   [0.0, 1.0] → [1.0]
   [1.0, 0.0] → [1.0]
   [1.0, 1.0] → [0.0]

🚀 Iniciando entrenamiento en paralelo...
   Learning rate: 0.1
   Épocas máximas: 10000
   Error objetivo: 0.01

ADAM: Época 0, Error = 0.575620
MOMENTUM: Época 0, Error = 0.508189
RMSPROP: Época 0, Error = 0.742123
🎉 RMSPROP CONVERGIÓ en época 80 con error 0.009616!
🎉 ADAM CONVERGIÓ en época 140 con error 0.009810!
MOMENTUM: Época 500, Error = 0.488309
MOMENTUM: Época 1000, Error = 0.014858
🎉 MOMENTUM CONVERGIÓ en época 1094 con error 0.009986!
...

📈 RESULTADOS FINALES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔹 RMSPROP → Error final: 0.009616 | Épocas: 81 | Tiempo: 3ms ✅
🔹 ADAM → Error final: 0.009810 | Épocas: 141 | Tiempo: 6ms ✅
🔹 MOMENTUM → Error final: 0.009986 | Épocas: 1095 | Tiempo: 26ms ✅

🏆 GANADOR: RMSPROP con error final 0.009616 en 81 épocas (3ms)
```



## 📈 Interpretación de Resultados

### Métricas Evaluadas
- **Error final**: Menor es mejor
- **Épocas hasta convergencia**: Menor indica convergencia más rápida
- **Tiempo de ejecución**: Menor es más eficiente
- **Convergencia**: Si alcanzó el error objetivo (< 0.01)

### Resultados Típicos
En general, sueles observar estos patrones:

1. **Adam**: Convergencia más rápida y estable
2. **RMSprop**: Buen balance entre velocidad y estabilidad
3. **Momentum**: Puede ser más lento pero a veces encuentra mejores mínimos

> **Nota**: Los resultados pueden variar según la semilla aleatoria y los hiperparámetros.

## 🔧 Personalización

### Modificar Hiperparámetros
En la función `main()`:
```rust
let learning_rate = 0.1;        // Tasa de aprendizaje
let max_epochs = 10000;         // Épocas máximas
let target_error = 0.01;        // Error objetivo
let seed = 42;                  // Semilla para reproducibilidad
```

### Modificar Arquitectura de Red
En `NeuralNetwork::new_with_seed()`:
```rust
let network = NeuralNetwork::new_with_seed(
    2,              // Neuronas de entrada
    4,              // Neuronas ocultas
    1,              // Neuronas de salida
    learning_rate,
    seed
);
```

### Modificar Parámetros de Optimizadores
```rust
// Momentum
let optimizer = MomentumOptimizer::new(&network, 0.9);  // β

// RMSprop  
let optimizer = RMSpropOptimizer::new(&network, 0.9, 1e-8);  // α, ε

// Adam
let optimizer = AdamOptimizer::new(&network, 0.9, 0.999, 1e-8);  // β₁, β₂, ε
```

## 📚 Conceptos Aprendidos

Este proyecto ilustra varios conceptos importantes:

- **Backpropagation**: Cálculo de gradientes mediante la regla de la cadena
- **Algoritmos de optimización**: Diferentes estrategias para actualizar pesos
- **Paralelismo en Rust**: Uso seguro de hilos y sincronización
- **Trait system**: Polimorfismo sin overhead de runtime
- **Memory management**: Ownership, borrowing y lifetimes



---

**¿Encontraste este proyecto útil?** ⭐ ¡Dale una estrella al repositorio!