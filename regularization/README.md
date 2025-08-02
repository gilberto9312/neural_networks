# 🧠 Regularización en Redes Neuronales desde Cero

## 📚 ¿Qué Aprendimos Hoy?

Implementamos y comparamos **tres técnicas fundamentales de regularización** en una red neuronal construida completamente desde cero en Rust:

- **L1 Regularization (Lasso)**: Penaliza con el valor absoluto de los pesos
- **L2 Regularization (Ridge)**: Penaliza con el cuadrado de los pesos  
- **Dropout**: Apaga aleatoriamente neuronas durante el entrenamiento

## 🔧 Arquitectura Modular

### Características Implementadas:
- ✅ **Regularizadores independientes** usando traits
- ✅ **Configuración flexible** (cada técnica por separado o combinada)
- ✅ **Sistema de convergencia** con detección automática de aprendizaje
- ✅ **Comparación automática** de rendimiento entre métodos
- ✅ **Sin dependencias externas** - todo implementado desde cero

### Estructura del Código:
```rust
// Trait genérico para regularización
trait Regularizer {
    fn compute_loss(&self, network: &NeuralNetwork) -> f64;
    fn apply_to_gradient(&self, gradient: f64, weight: f64) -> f64;
    fn name(&self) -> &str;
}

// Implementaciones específicas
struct L1Regularizer { lambda: f64 }
struct L2Regularizer { lambda: f64 }  
struct DropoutRegularizer { rate: f64 }
```

## 🔬 Experimento: Función XOR

### Datos de Entrenamiento:
```
[0.0, 0.0] → [0.0]
[0.0, 1.0] → [1.0] 
[1.0, 0.0] → [1.0]
[1.0, 1.0] → [0.0]
```

### Configuración:
- **Red**: 2 entradas → 8 ocultas → 1 salida
- **Optimizador**: Adam (β1=0.9, β2=0.999)
- **Learning Rate**: 0.1
- **Error Objetivo**: 0.01
- **Épocas Máximas**: 10,000

## 📊 Resultados Experimentales

### Primera Prueba (λ = 0.01):
```
📈 RESUMEN FINAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔹 Sin regularización ✅ → Épocas: 109, Val Error: 0.009926
🔹 L1 (λ=0.01) ❌ → Épocas: 10000, Val Error: 0.500003
🔹 L2 (λ=0.01) ❌ → Épocas: 10000, Val Error: 0.500003
🔹 Dropout (50%) ✅ → Épocas: 733, Val Error: 0.009196
🔹 L2 + Dropout ❌ → Épocas: 10000, Val Error: 0.500003
```

### Resultados Optimizados (λ ajustado):
```
📈 RESUMEN FINAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔹 Sin regularización ✅ → Épocas: 109, Val Error: 0.009926
🔹 L1 (λ=0.001) ✅ → Épocas: 185, Val Error: 0.009954
🔹 L2 (λ=0.001) ✅ → Épocas: 195, Val Error: 0.009997
🔹 Dropout (50%) ✅ → Épocas: 733, Val Error: 0.009196
🔹 L2 + Dropout ✅ → Épocas: 751, Val Error: 0.009165
```

## 🧐 Análisis de Resultados

### 🔍 Observaciones Clave:

#### **1. Regularización L1/L2 con λ = 0.01:**
- ❌ **Completamente bloquea el aprendizaje**
- **Problema**: Lambda demasiado alto para una red pequeña
- **Error constante**: ~0.5 (equivale a predicciones aleatorias)
- **Lección**: En problemas simples, regularización excesiva mata el aprendizaje

#### **2. Regularización L1/L2 con λ = 0.001:**
- ✅ **Aprenden correctamente** 
- **Ligeramente más lento** que sin regularización (185-195 vs 109 épocas)
- **Error final similar** al baseline
- **Lección**: El hiperparámetro λ es crítico

#### **3. Dropout (50%):**
- ✅ **Funciona excelente**
- **Más robusto** que regularización de pesos
- **Mejor generalización** (menor error de validación: 0.009196)
- **Costo**: Requiere más épocas (733 vs 109)

#### **4. L2 + Dropout:**
- ✅ **Mejor resultado general** (error: 0.009165)
- **Combinación ganadora** cuando está bien ajustada
- **Convergencia estable** aunque más lenta

### 📈 Ranking de Técnicas:
1. **🥇 L2 + Dropout**: Mejor generalización (0.009165)
2. **🥈 Dropout solo**: Segundo mejor error (0.009196)  
3. **🥉 Sin regularización**: Más rápido pero menos robusto (109 épocas)
4. **L2 solo**: Funciona pero sin ventajas claras
5. **L1 solo**: Funciona pero tiende a ser más agresivo

## 💡 Lecciones Aprendidas

### **Técnicas de Regularización:**

#### **L1 Regularization:**
- **Matemática**: `Loss += λ * Σ|w|`
- **Gradiente**: `grad += λ * sign(w)`
- **Efecto**: Fuerza pesos a exactamente 0 → **selección automática de características**
- **Uso**: Cuando quieres un modelo esparso

#### **L2 Regularization:**
- **Matemática**: `Loss += λ * Σw²`  
- **Gradiente**: `grad += λ * w`
- **Efecto**: Pesos pequeños pero no cero → **modelo más suave**
- **Uso**: Prevención general de overfitting

#### **Dropout:**
- **Durante entrenamiento**: Apaga neuronas con probabilidad `p`
- **Durante evaluación**: Todas activas, escaladas por `(1-p)`
- **Efecto**: Previene co-adaptación → **mejor generalización**
- **Uso**: Especialmente efectivo en redes grandes

### **Hiperparámetros Críticos:**
- **L1/L2 λ**: 0.001 - 0.01 (¡muy sensible!)
- **Dropout rate**: 0.2 - 0.5 para capas ocultas
- **Combinaciones**: Reducir λ cuando se combina con dropout

### **Cuándo Usar Cada Técnica:**
- **Problemas simples**: Dropout > L1/L2 (menos sensible a hiperparámetros)
- **Selección de características**: L1
- **Suavizado general**: L2
- **Redes grandes**: Dropout + L2 (combinación ganadora)
- **Recursos limitados**: Sin regularización puede ser suficiente

### **¿por qué adam se puedecombinar con l2  y no con RMSPROP?**

| Técnica   | ¿Qué hace?                                                                 |
|-----------|-----------------------------------------------------------------------------|
| **L2**    | Penaliza pesos grandes, agregando `λ * w²` al costo para evitar overfitting. |
| **RMSprop** | Ajusta la tasa de aprendizaje por parámetro, según la media cuadrática de los gradientes. |
| **Adam**  | Es como RMSprop + Momentum. Guarda momentos de primer y segundo orden.     |

---

### 🎯 ¿Qué significa "combinar con L2"?

Cuando se entrena una red neuronal, puedes aplicar **L2 regularization** de dos formas:

- **L2 clásica** (weight decay explícito): se agrega `λ * w` al gradiente durante backpropagation.
- **Weight decay implícito**: se escala el peso después del paso de optimización (técnica más moderna y precisa en ciertos contextos).

---

### 💥 El problema con RMSProp + L2

RMSprop ajusta el denominador del gradiente, dividiéndolo por la raíz cuadrada de una media móvil de los gradientes al cuadrado:

```math
w = w - η * (grad / sqrt(E[g²]) + ε)
```

Si a ese `grad` le agregas `λ * w` (la parte L2), el optimizador la trata como parte del gradiente y la divide también por `sqrt(E[g²])`, lo cual **distorsiona la regularización**.

> En otras palabras: RMSprop **escala la penalización L2 de forma inadecuada e inconsistente**.

---

### ✅ ¿Por qué Adam sí funciona con L2?

Adam hace lo mismo que RMSprop, pero también:

- Corrige el sesgo de los promedios móviles
- Aplica `weight decay` correctamente si se usa la forma **moderna** (como en **AdamW**)

### Forma clásica con L2

```rust
// En pseudo-Rust:
grad = grad + λ * w; // L2 regularization
m = β1 * m + (1 - β1) * grad
v = β2 * v + (1 - β2) * grad^2
w = w - lr * (m / sqrt(v) + ε)
```

### Forma moderna con AdamW (decoupled weight decay)

```rust
// AdamW (decoupled weight decay)
w = w * (1 - lr * λ) - lr * (m / sqrt(v) + ε)
```

> Así el peso se decae **independientemente del gradiente**. ¡Y esa es la forma moderna recomendada!

---

### 🧪 En resumen

| Técnica   | ¿Combina bien con L2? | ¿Por qué? |
|-----------|------------------------|-----------|
| **RMSprop** | ❌ No recomendado | La L2 es mal escalada al dividirse con `sqrt(E[g²])`. |
| **Adam**    | ✅ Sí               | Puede usarse con L2 clásico o **AdamW** (mejor), aplicando decay correctamente. |


## 🚀 Implementación Práctica

### Uso Básico:
```rust
// Sin regularización
let network = NeuralNetwork::new_with_seed(2, 8, 1, 0.1, 42);

// Solo L1  
let l1_reg = L1Regularizer::new(0.001);
network.compute_gradients(inputs, targets, Some(&l1_reg));

// Solo L2
let l2_reg = L2Regularizer::new(0.001);
network.compute_gradients(inputs, targets, Some(&l2_reg));

// Solo Dropout
let network = NeuralNetwork::new_with_seed(2, 8, 1, 0.1, 42)
    .with_dropout(0.5);

// L2 + Dropout (combinación ganadora)
let network = NeuralNetwork::new_with_seed(2, 8, 1, 0.1, 42)
    .with_dropout(0.3);
let l2_reg = L2Regularizer::new(0.001);
```

### Sistema de Convergencia:
```rust
// Detecta automáticamente cuándo la red "aprendió"
if val_error < target_error {
    println!("🎉 {} CONVERGIÓ en época {} con error {:.6}!", 
             name, epoch, val_error);
    break;
}
```




## 🎯 Conclusiones

1. **La regularización no siempre mejora**: En problemas simples puede ser contraproducente
2. **Los hiperparámetros son críticos**: Un λ incorrecto mata completamente el aprendizaje  
3. **Dropout es muy robusto**: Menos sensible a configuración que L1/L2
4. **Las combinaciones pueden ser ganadoras**: L2 + Dropout obtuvo el mejor resultado
5. **La implementación modular paga**: Facilita enormemente la experimentación



---

*Construido completamente desde cero en Rust 🦀 - Sin dependencias externas - Enfoque educativo*