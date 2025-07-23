# Día 5: Perceptrón Entrenable desde Cero 🧠

**Reto de 21 Días - Redes Neuronales desde Cero**

## 🎯 Objetivo del Día

Implementar un **perceptrón de una sola neurona** completamente desde cero en Rust, sin librerías externas. El objetivo es entender los conceptos fundamentales del aprendizaje automático:

- Cómo una neurona artificial procesa información
- Cómo ajusta sus parámetros para aprender
- Cómo resolver problemas de clasificación binaria

## 🧮 Conceptos Aprendidos

### 1. Arquitectura del Perceptrón
- **Entradas**: Valores que recibe la neurona (x₁, x₂, ...)
- **Pesos**: Parámetros que se ajustan durante el entrenamiento (w₁, w₂, ...)
- **Bias**: Parámetro que permite desplazar la función de decisión
- **Función de activación**: Función escalón que produce la salida binaria

### 2. Matemáticas Fundamentales

**Forward Pass (Predicción):**
```
z = w₁×x₁ + w₂×x₂ + ... + bias
salida = step_function(z) = { 1 si z ≥ 0
                            { 0 si z < 0
```

**Regla del Perceptrón (Aprendizaje):**
```
error = salida_esperada - salida_obtenida
nuevo_peso = peso_actual + (tasa_aprendizaje × error × entrada)
nuevo_bias = bias_actual + (tasa_aprendizaje × error)
```

### 3. Proceso de Entrenamiento
1. Inicializar pesos con valores aleatorios pequeños
2. Para cada ejemplo de entrenamiento:
   - Hacer predicción (forward pass)
   - Calcular error
   - Ajustar pesos y bias
3. Repetir hasta convergencia

## 🚀 Cómo Ejecutar

```bash
# Compilar y ejecutar
cargo run

# O directamente con rustc
rustc main.rs -o perceptron
./perceptron
```

## 📊 Problema Resuelto: Compuerta AND

El perceptrón aprende a implementar una compuerta lógica AND:

| Entrada 1 | Entrada 2 | Salida |
|-----------|-----------|--------|
| 0         | 0         | 0      |
| 0         | 1         | 0      |
| 1         | 0         | 0      |
| 1         | 1         | 1      |

## 🔧 Estructura del Código

```rust
struct Perceptron {
    weights: Vec<f64>,      // Pesos de las conexiones
    bias: f64,              // Término de sesgo
    learning_rate: f64,     // Velocidad de aprendizaje
}
```

**Métodos principales:**
- `new()`: Constructor con inicialización aleatoria
- `predict()`: Realiza una predicción (forward pass)
- `train_step()`: Entrena con un solo ejemplo
- `train()`: Entrena con múltiples ejemplos durante varias épocas

## 🧪 Experimentos Sugeridos

### 1. Cambiar la Tasa de Aprendizaje
```rust
let mut perceptron = Perceptron::new(2, 0.01); // Más lento
let mut perceptron = Perceptron::new(2, 0.5);  // Más rápido
let mut perceptron = Perceptron::new(2, 1.0);  // Muy rápido (¿inestable?)
```

### 2. Probar con Compuerta OR
Descomenta la función `create_training_data_or()` y úsala en lugar de los datos de AND.

### 3. Intentar con XOR (Spoiler: No Funcionará)
```rust
let training_data = vec![
    (vec![0.0, 0.0], 0.0), // 0 XOR 0 = 0
    (vec![0.0, 1.0], 1.0), // 0 XOR 1 = 1
    (vec![1.0, 0.0], 1.0), // 1 XOR 0 = 1
    (vec![1.0, 1.0], 0.0), // 1 XOR 1 = 0
];
```

**¿Por qué no funciona XOR?** Porque no es linealmente separable. Necesitarías múltiples neuronas (próximos días).

### 4. Agregar Ruido a los Datos
Usa la función `add_noise_to_data()` para ver cómo se comporta el perceptrón con datos imperfectos.

## 📈 Salida Esperada

```
=== Entrenando Perceptrón para compuerta AND ===

Estado inicial:
Pesos: [-0.123, 0.456]
Bias: 0.100

Datos de entrenamiento (compuerta AND):
[0.0, 0.0] -> 0
[0.0, 1.0] -> 0
[1.0, 0.0] -> 0
[1.0, 1.0] -> 1

Época 0: Error total = 2.00
Época 10: Error total = 0.00
¡Perceptrón entrenado perfectamente en 15 épocas!

=== Probando el perceptrón entrenado ===
Entrada: [0.0, 0.0] -> Predicción: 0, Esperado: 0 ✓
Entrada: [0.0, 1.0] -> Predicción: 0, Esperado: 0 ✓
Entrada: [1.0, 0.0] -> Predicción: 0, Esperado: 0 ✓
Entrada: [1.0, 1.0] -> Predicción: 1, Esperado: 1 ✓
```

## 🤔 Preguntas y Respuestas Fundamentales

### 1. ¿Por qué inicializamos los pesos con valores aleatorios pequeños?

**Valores aleatorios** son necesarios porque:
- Si todos los pesos empiezan iguales (ej: todos en 0), la neurona no puede "romper la simetría"
- Cada peso debe evolucionar de forma independiente para especializarse
- La aleatoriedad permite explorar diferentes soluciones iniciales

**Valores pequeños** son importantes porque:
- Pesos grandes pueden saturar la función de activación desde el inicio
- Empezar cerca de cero permite ajustes graduales y suaves
- Evita que la neurona "se convenza" demasiado pronto de una decisión incorrecta

**Ejemplo**: Si inicializáramos todos los pesos en 10.0, casi cualquier entrada daría una suma muy grande, haciendo que la neurona siempre prediga 1, dificultando el aprendizaje.

### 2. ¿Qué pasaría si la tasa de aprendizaje fuera muy grande o muy pequeña?

**Tasa muy pequeña (ej: 0.001)**:
- ✅ Convergencia estable y suave
- ❌ Aprendizaje extremadamente lento
- ❌ Puede quedarse atascado en mínimos locales
- ❌ Necesita muchas más épocas

**Tasa muy grande (ej: 2.0)**:
- ✅ Aprendizaje rápido inicialmente
- ❌ Oscilaciones alrededor de la solución óptima
- ❌ Puede "saltar" sobre la solución correcta
- ❌ Inestabilidad: los pesos pueden crecer descontroladamente

**Tasa equilibrada (ej: 0.1-0.5)**:
- ✅ Balance entre velocidad y estabilidad
- ✅ Convergencia en pocas épocas
- ✅ Ajustes controlados

### 3. ¿Cómo podríamos visualizar la línea de decisión que crea el perceptrón?

El perceptrón crea una **línea de decisión** definida por la ecuación:
```
w₁×x₁ + w₂×x₂ + bias = 0
```

**Opción 1: Visualización con caracteres**
```rust
fn visualizar_decision_2d(perceptron: &Perceptron) {
    println!("Mapa de decisión (0=Negro, 1=Blanco):");
    for y in 0..10 {
        for x in 0..10 {
            let input = vec![x as f64 / 9.0, y as f64 / 9.0];
            let prediction = perceptron.predict(&input);
            print!("{} ", if prediction == 1.0 { "⬜" } else { "⬛" });
        }
        println!();
    }
}
```

**Opción 2: Ecuación matemática de la línea**
```rust
fn mostrar_linea_decision(perceptron: &Perceptron) {
    let w1 = perceptron.weights[0];
    let w2 = perceptron.weights[1];
    let b = perceptron.bias;
    
    println!("Línea de decisión: {:.3}×x₁ + {:.3}×x₂ + {:.3} = 0", w1, w2, b);
    
    // Para x₁ = 0: x₂ = -bias/w₂
    // Para x₁ = 1: x₂ = -(w₁ + bias)/w₂
    if w2 != 0.0 {
        let y_cuando_x0 = -b / w2;
        let y_cuando_x1 = -(w1 + b) / w2;
        println!("Línea pasa por: (0, {:.3}) y (1, {:.3})", y_cuando_x0, y_cuando_x1);
    }
}
```

### 4. ¿Qué otros problemas linealmente separables podrías resolver?

**Compuertas lógicas simples**:
- **OR**: Al menos una entrada debe ser 1
- **NOR**: Ninguna entrada debe ser 1 (NOT OR)  
- **NAND**: No ambas entradas pueden ser 1 (NOT AND)

**Problemas de clasificación binaria del mundo real**:
```rust
// Ejemplo: Clasificar personas como "altas y pesadas"
let datos_altura_peso = vec![
    (vec![1.6, 60.0], 0.0), // Bajo, ligero -> 0
    (vec![1.9, 90.0], 1.0), // Alto, pesado -> 1  
    (vec![1.5, 50.0], 0.0), // Bajo, ligero -> 0
    (vec![1.8, 85.0], 1.0), // Alto, pesado -> 1
];

// Ejemplo: Decidir si aprobar un préstamo
let datos_prestamo = vec![
    (vec![25000.0, 650.0], 0.0), // Salario bajo, mal crédito -> No
    (vec![60000.0, 750.0], 1.0), // Salario alto, buen crédito -> Sí
    (vec![30000.0, 600.0], 0.0), // Límites -> No
    (vec![80000.0, 720.0], 1.0), // Buenos números -> Sí
];

// Ejemplo: Detección de spam simple
let datos_spam = vec![
    (vec![10.0, 2.0], 1.0),  // Muchos números, pocas palabras -> Spam
    (vec![2.0, 15.0], 0.0),  // Pocos números, muchas palabras -> No spam
    (vec![8.0, 3.0], 1.0),   // Muchos números -> Spam
    (vec![1.0, 12.0], 0.0),  // Texto normal -> No spam
];
```

**Detección de patrones geométricos simples**:
- Detectar si un punto está arriba o abajo de una línea diagonal
- Clasificar temperaturas como "frío/calor" basado en temperatura y humedad
- Determinar si un estudiante pasará (horas de estudio + calificación previa)

**⚠️ Importante - Lo que el perceptrón NO puede resolver**:
- **XOR** (exclusivo o) - No es linealmente separable
- **Problemas no lineales** (círculos, espirales, formas curvas)
- **Clasificación multi-clase** directa (necesita múltiples perceptrones)

La clave está en que debe existir una **línea recta** que pueda separar perfectamente las dos clases en el espacio de características.

## 🎓 Lo Que Hemos Logrado

- ✅ Implementación completa de un perceptrón desde cero
- ✅ Comprensión del proceso de aprendizaje supervisado
- ✅ Experiencia práctica con forward pass y backpropagation básica
- ✅ Fundamentos matemáticos para redes neuronales más complejas

---
**¡Día 5 completado! 🎉 Solo quedan 16 días más hacia el dominio de las redes neuronales PD: si el commit con el dia 4 es el mismo es porque el del dia 4 estuvo heavy.**