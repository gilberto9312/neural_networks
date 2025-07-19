# 🧠 Redes Neuronales - Serie de Aprendizaje Diario

## 📅 Día 2: Función Sigmoid

### 🎯 Objetivo del Día
Entender cómo la función sigmoid transforma las salidas lineales de una neurona en salidas no-lineales, y por qué esto es fundamental para el funcionamiento de las redes neuronales.

### 🔍 ¿Qué es la Función Sigmoid?

La función sigmoid es una función matemática que toma cualquier número real y lo "comprime" a un valor entre 0 y 1. Su fórmula es:

```
σ(z) = 1 / (1 + e^(-z))
```

#### Propiedades clave:
- **Rango**: (0, 1) - nunca llega exactamente a 0 o 1
- **Forma**: Curva suave en forma de "S"
- **Punto medio**: σ(0) = 0.5
- **Asíntotas**: Cuando z → +∞, σ(z) → 1; cuando z → -∞, σ(z) → 0
- **Derivable**: En todos los puntos, lo que permite el entrenamiento por gradiente

### 🏗️ Estructura del Código

#### `struct Neuron`
Representa una neurona artificial con:
- `synapses`: Vector de pesos sinápticos (conexiones)
- `threshold`: Umbral o bias de la neurona

#### Métodos principales:
1. **`salida_lineal()`**: Calcula z = Σ(wi * xi) + b
2. **`salida_con_sigmoid()`**: Aplica sigmoid a la salida lineal
3. **`sigmoid()`**: Implementa la función matemática sigmoid

### 🧪 Experimentos Realizados

#### 1. Análisis Teórico
- Exploración de las propiedades matemáticas de sigmoid
- Prueba con valores desde -10 hasta +10
- Observación de cómo los valores extremos se saturan

#### 2. Comparación Práctica
- **Salida Lineal**: Puede ser cualquier valor real (negativo, positivo, grande)
- **Salida Sigmoid**: Siempre entre 0 y 1, interpretable como probabilidad

### 💡 Conceptos Clave Aprendidos

1. **Transformación no-lineal**: Sigmoid convierte la combinación lineal en una salida no-lineal
2. **Saturación**: Valores muy grandes o muy pequeños se "saturan" cerca de 1 o 0
3. **Suavidad**: La curva continua permite calcular gradientes para el entrenamiento
4. **Interpretabilidad**: La salida puede interpretarse como una probabilidad
5. **Activación**: Determina cuándo la neurona "se activa" (valores cercanos a 1)

### 🔧 Cómo Ejecutar

```bash
# Compilar y ejecutar
cargo run

# O si usas rustc directamente
rustc main.rs && ./main
```

### 📊 Ejemplo de Salida

```
🧠 REDES NEURONALES - DÍA: FUNCIÓN SIGMOID
===========================================

📊 COMPARACIÓN: LINEAL vs NO-LINEAL
Entradas             | Salida Lineal  | Salida Sigmoid
-------------------------------------------------------
[1.0, 2.0, -1.0]    |     -0.1000    |     0.475021
[-2.0, 1.5, 3.0]    |     1.1500     |     0.759469
[0.0, 0.0, 0.0]     |     0.1000     |     0.524979
[10.0, -5.0, 2.0]   |     8.2000     |     0.999725
```

### 🎓 ¿Por qué es Importante Sigmoid?

1. **No-linealidad**: Sin funciones de activación como sigmoid, una red neuronal sería solo una transformación lineal, sin importar cuántas capas tenga.

2. **Gradientes**: La función es diferenciable, permitiendo el algoritmo de backpropagation para entrenar la red.

3. **Interpretación**: Los valores entre 0 y 1 pueden interpretarse como probabilidades en problemas de clasificación binaria.

4. **Control de activación**: Determina cuándo y cuánto se "activa" una neurona basándose en sus entradas.


### 📝 Notas de Desarrollo

Este es el **Día 2** de una serie de aprendizaje progresivo sobre redes neuronales. El código está diseñado para ser educativo, con:
- Comentarios detallados en español
- Salida explicativa paso a paso
- Ejemplos prácticos con diferentes entradas
- Comparaciones directas entre enfoques lineales y no-lineales

---

*Implementado en Rust para practicar tanto conceptos de ML como programación de sistemas.*