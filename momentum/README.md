# 🚀 Día 8: Implementando Momentum Optimizer desde Cero

## 📝 Resumen del Día

Hoy implementamos el optimizador **Momentum** en nuestra red neuronal multicapa, reemplazando el SGD básico. El objetivo era entender a fondo cómo funciona Momentum sin usar librerías externas, solo tipos nativos de Rust.

## 🎯 Objetivos Cumplidos

- ✅ Comprender el concepto de Momentum conceptualmente
- ✅ Implementar estructuras de datos para almacenar velocidades
- ✅ Modificar el algoritmo de actualización de pesos
- ✅ Debuggear problemas de convergencia
- ✅ Entender las convenciones matemáticas correctas

## 🧠 ¿Qué es Momentum?

### Problema con SGD básico:
```
peso = peso - learning_rate * gradiente
```
- Solo considera el gradiente actual
- Puede oscilar en valles estrechos
- Se queda atrapado en mínimos locales

### Solución con Momentum:
```
velocidad = β * velocidad_anterior + gradiente
peso = peso - learning_rate * velocidad
```
- Simula "inercia" física
- Suaviza oscilaciones
- Ayuda a escapar de mínimos locales
- Acelera convergencia en direcciones consistentes

## 🏗️ Implementación en Rust

### 1. Estructuras de Datos Adicionales

```rust
struct NeuralNetwork {
    // Pesos originales
    weights_input_hidden: Vec<Vec<f64>>,
    weights_hidden_output: Vec<Vec<f64>>,
    bias_hidden: Vec<f64>,
    bias_output: Vec<f64>,
    
    // ¡NUEVAS! - Velocidades para Momentum
    velocity_weights_input_hidden: Vec<Vec<f64>>,
    velocity_weights_hidden_output: Vec<Vec<f64>>,
    velocity_bias_hidden: Vec<f64>,
    velocity_bias_output: Vec<f64>,
    
    learning_rate: f64,
    momentum: f64,  // Factor β (típicamente 0.8-0.9)
}
```

### 2. Función de Actualización

```rust
fn update_with_momentum(
    weight: &mut f64, 
    velocity: &mut f64, 
    gradient: f64, 
    learning_rate: f64, 
    momentum: f64
) {
    // Actualizar velocidad
    *velocity = momentum * (*velocity) + gradient;
    
    // Actualizar peso usando velocidad
    *weight -= learning_rate * (*velocity);
}
```

### 3. Inicialización

- **Pesos:** Inicializados con valores pequeños aleatorios
- **Velocidades:** Todas inicializadas en `0.0`

## 🐛 Problemas Encontrados y Soluciones

### Problema 1: Borrow Checker de Rust
**Error:** `cannot borrow *self as mutable more than once`

**Solución:** Hacer la función `update_with_momentum` estática:
```rust
// ❌ Antes:
fn update_with_momentum(&mut self, weight: &mut f64, ...)

// ✅ Después:
fn update_with_momentum(weight: &mut f64, velocity: &mut f64, ...)
```

### Problema 2: Red No Convergía
**Síntomas:** Todas las salidas cerca de 0, red no aprendía XOR

**Diagnóstico:**
- Pesos muy negativos (-2.4 a -2.0)
- Velocidades microscópicas (0.0001)

**Causa raíz:** Error en la definición del gradiente

### Problema 3: Convención Matemática Incorrecta
**El problema:**
```rust
// Nuestro "delta"
let delta = error * out * (1.0 - out);  // donde error = target - output
```

**Esto NO es el gradiente real.** El gradiente de la función de pérdida `L = 0.5(target - output)²` es:
```
∂L/∂w = -(target - output) * output * (1-output) * input
```

**Solución:** Agregar el signo negativo:
```rust
let delta = -error * out * (1.0 - out);  // Ahora SÍ es el gradiente
```

## ⚙️ Parámetros Óptimos Encontrados

```rust
let mut network = NeuralNetwork::new(
    2,    // input_size
    4,    // hidden_size
    1,    // output_size
    0.1,  // learning_rate (reducido de 0.5)
    0.8   // momentum (reducido de 0.9)
);
```

## 📊 Resultados

**Antes (SGD básico):** Convergencia lenta, posibles oscilaciones
**Después (Momentum):** Convergencia más suave y potencialmente más rápida

**Función XOR aprendida correctamente:**
```
Entrada: [0.0, 0.0] → Salida: 0.0001, Predicción: 0, Esperado: 0, ✓
Entrada: [0.0, 1.0] → Salida: 0.9999, Predicción: 1, Esperado: 1, ✓
Entrada: [1.0, 0.0] → Salida: 0.9999, Predicción: 1, Esperado: 1, ✓
Entrada: [1.0, 1.0] → Salida: 0.0001, Predicción: 0, Esperado: 0, ✓
```

## 🔍 Herramientas de Debug Implementadas

```rust
fn debug_network_state(&self) {
    // Estadísticas de pesos
    println!("Pesos oculta→salida: min={:.4}, max={:.4}, promedio={:.4}");
    
    // Estadísticas de velocidades
    println!("Velocidades (abs): min={:.6}, max={:.6}, promedio={:.6}");
}
```

## 📚 Conceptos Clave Aprendidos

1. **Momentum = Inercia:** Los pesos "recuerdan" la dirección de actualizaciones previas
2. **Factor β:** Controla cuánta historia mantener (0.8-0.9 típico)
3. **Rust Borrow Checker:** Evitar múltiples referencias mutables usando funciones estáticas
4. **Gradientes vs Deltas:** Importancia de las convenciones matemáticas correctas
5. **Debugging:** Monitorear rangos de pesos y velocidades para diagnosticar problemas

## 🔄 Fórmula Final (Correcta)

```
gradiente = -(target - output) * output * (1-output) * input
velocidad = β * velocidad_anterior + gradiente  
peso = peso - learning_rate * velocidad
```


## 📖 Recursos para Profundizar

- Paper original de Momentum: Rumelhart et al. (1986)
- Convenciones de gradientes en deep learning
- Optimizadores modernos: Adam, RMSprop, AdaGrad

---

**Lecciones del día:** La implementación desde cero enseña detalles cruciales que las librerías ocultan. Los pequeños errores de convención pueden tener grandes impactos, y el debugging sistemático es clave para el aprendizaje profundo.

*¡Día 7 completado exitosamente! 🎉*