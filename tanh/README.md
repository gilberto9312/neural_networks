# 🧠 Redes Neuronales: Tanh vs Sigmoid en Rust

## 📚 ¿Qué hemos aprendido hoy?

Este proyecto demuestra de manera práctica las diferencias fundamentales entre las funciones de activación **Sigmoid** y **Tanh** en redes neuronales, implementadas en Rust con explicaciones detalladas paso a paso.

## 🎯 Objetivos del Proyecto

- ✅ Implementar funciones de activación Sigmoid y Tanh
- ✅ Calcular sus derivadas para el proceso de backpropagation
- ✅ Comparar su comportamiento con ejemplos prácticos
- ✅ Entender el problema del **gradiente desvaneciente**
- ✅ Demostrar por qué el centrado en cero es importante

## 📊 Funciones de Activación

### 🔸 Sigmoid
```
σ(x) = 1 / (1 + e^(-x))
```
- **Rango:** (0, 1)
- **Características:** Siempre positiva, centrada en 0.5
- **Uso principal:** Clasificación binaria (capa de salida)

### 🔸 Tanh (Tangente Hiperbólica)
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```
- **Rango:** (-1, 1)
- **Características:** Centrada en 0, simétrica
- **Uso principal:** Capas ocultas

## 🧨 ¿Qué es el *Gradiente Desvaneciente*?

Es cuando los **gradientes se hacen tan pequeños** (cercanos a cero) que **dejan de actualizar los pesos** de las capas anteriores. Es como decirle a la red: *"Estás aprendiendo tan poquito que ya no vale la pena seguir."*

## 🧮 ¿Por qué pasa esto con `sigmoid` y `tanh`?

### 🔹 1. **Por su forma: se aplastan en los extremos**

Mira las curvas de `sigmoid(x)` y `tanh(x)`:
* **Sigmoid(x)** va de 0 a 1
* **Tanh(x)** va de -1 a 1

Ambas tienen una forma en S, y se "aplanan" en los extremos. Esto significa que:
* Cuando `x` es muy grande o muy pequeño, los valores de la derivada (el gradiente) son muy cercanos a **0**
* Y si la derivada es casi 0, los **pesos casi no cambian** al entrenar

### 🔹 2. **La derivada de sigmoid(x)**
```
d/dx sigmoid(x) = sigmoid(x)(1 - sigmoid(x))
```
📉 La derivada es máxima en `x = 0` (valor 0.25), pero **se va acercando a 0 rápidamente** cuando `x` se aleja de 0.

### 🔹 3. **La derivada de tanh(x)**
```
d/dx tanh(x) = 1 - tanh²(x)
```
📉 También se aplana: cuando `tanh(x)` ≈ ±1, la derivada ≈ 0.

## 📉 ¿Qué pasa en una red neuronal profunda?

Cuando tienes muchas capas y usas `sigmoid` o `tanh` en cada una:

1. Los gradientes se multiplican por valores muy pequeños repetidamente
2. **Resultado:** los gradientes se "desvanecen" antes de llegar a las capas del inicio
3. **Consecuencia:** si no hay gradiente... **no hay aprendizaje** en las capas profundas

Esto es lo que se llama el **problema del gradiente desvaneciente**.

## 🔄 Ejemplo Visual del Problema

```
Capa 5 (salida): gradiente = 0.3
Capa 4: 0.3 × 0.2 = 0.06  ← Ya se está reduciendo
Capa 3: 0.06 × 0.15 = 0.009  ← Muy pequeño
Capa 2: 0.009 × 0.1 = 0.0009  ← Casi cero
Capa 1: 0.0009 × 0.05 = 0.000045  ← ¡Desvanecido!
```

## 💡 Comparación Práctica

| Aspecto | Sigmoid | Tanh | Ganador |
|---------|---------|------|---------|
| **Rango** | (0, 1) | (-1, 1) | Tanh |
| **Centrado en cero** | ❌ No | ✅ Sí | Tanh |
| **Gradientes fuertes** | ❌ Débiles | ✅ Más fuertes | Tanh |
| **Clasificación binaria** | ✅ Perfecto | ❌ No ideal | Sigmoid |
| **Capas ocultas** | ❌ Problemático | ✅ Mejor | Tanh |
| **Velocidad de convergencia** | ❌ Lenta | ✅ Más rápida | Tanh |


## 📈 Resultados de las Demostraciones

### 1. **Comparación de Valores**
El programa muestra cómo cada función responde a diferentes entradas desde -3.0 hasta 3.0.

### 2. **Red Neuronal Simple**
Demuestra cómo las mismas entradas producen diferentes distribuciones:
- **Sigmoid:** valores entre 0.05 y 0.92
- **Tanh:** valores entre -0.96 y 0.92 (más distribuidos)

### 3. **Demostración de Centrado en Cero**
Explica por qué las funciones centradas en cero facilitan el entrenamiento.

## 🎯 Recomendaciones Prácticas

### ✅ **Usa TANH cuando:**
- Trabajas con capas ocultas
- Necesitas gradientes más fuertes
- Quieres convergencia más rápida
- Los datos están normalizados

### ✅ **Usa SIGMOID cuando:**
- Necesitas salida de probabilidad (0-1)
- Clasificación binaria en la capa de salida
- Interpretar como "porcentaje de activación"

### ⚡ **En la práctica moderna:**
- **ReLU** y sus variantes son más populares para capas ocultas
- **Sigmoid** sigue siendo útil para capas de salida
- **Tanh** aún se usa en LSTMs y algunos casos específicos

## 🧪 Experimentos que puedes hacer

1. **Cambiar los valores de entrada:** Modifica el vector `test_values` para ver comportamientos extremos
2. **Implementar ReLU:** Agrega `relu(x) = max(0, x)` y compara
3. **Red más profunda:** Simula el gradiente desvaneciente con más capas
4. **Visualización:** Usa un graficador para ver las curvas

## 📚 Conceptos Clave Aprendidos

- **Función de activación:** Transforma la suma ponderada en una salida no lineal
- **Derivada:** Necesaria para calcular gradientes en backpropagation
- **Gradiente desvaneciente:** Problema donde los gradientes se vuelven muy pequeños
- **Centrado en cero:** Característica que mejora el entrenamiento
- **Saturación:** Cuando la función se "aplana" y pierde sensibilidad

## 🔗 Recursos Adicionales

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Understanding Activation Functions](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html)
- [The Vanishing Gradient Problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)

