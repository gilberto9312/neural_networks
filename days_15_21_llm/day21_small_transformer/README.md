# 🚀 Small Transformer LLM - Día 21

**Proyecto Final del Plan de 21 Días de Aprendizaje de LLMs**

Implementación educativa completa de un Transformer Language Model (decoder-only, estilo GPT) desde cero en Rust, sin usar librerías de ML de alto nivel.

## 📋 Descripción del Proyecto

Este es el proyecto culminante del plan de 21 días, integrando todos los conceptos aprendidos:
- **Días 15-16**: N-grams y Perplexity
- **Día 17**: Tokenización (BPE)
- **Día 18**: Word Embeddings
- **Día 19**: MLPs para Texto
- **Día 20**: Mecanismo de Atención
- **Día 21**: **Transformer Completo** ← ESTÁS AQUÍ

## ✅ Características Implementadas

- ✅ **Multi-Head Attention** con máscara causal
- ✅ **Positional Encoding** sinusoidal
- ✅ **Feed-Forward Networks** con activación ReLU
- ✅ **Layer Normalization**
- ✅ **Residual Connections**
- ✅ **Decoder Stack** (2 capas)
- ✅ **Embedding Layer**
- ✅ **Generación autoregressiva** de texto
- ✅ **Tokenizer simple** basado en palabras
- ✅ **Loop de entrenamiento** (forward pass + métricas)
- ✅ **Checkpoint saving/loading** (configuración y tokenizer)
- ✅ **CLI funcional** con múltiples comandos

## 🏗️ Arquitectura del Modelo

```
TransformerLM (Decoder-Only, GPT-style)
├── Embedding Layer (vocab_size → d_model)
├── Positional Encoding (sinusoidal)
├── Decoder Stack (2 capas)
│   ├── Masked Multi-Head Attention (4 cabezas)
│   ├── Layer Normalization
│   ├── Feed-Forward Network (d_model → 4*d_model → d_model)
│   └── Layer Normalization
└── Output Projection (d_model → vocab_size)
```

### Parámetros del Modelo

- **Vocab size**: ~30 tokens (simplificado para demostración)
- **Embedding dim**: 128
- **Num heads**: 4
- **Num layers**: 2 (decoder)
- **d_ff**: 512 (4 × d_model)
- **Max seq len**: 64
- **Total params**: ~403,200

## 🚀 Cómo Ejecutar

### Compilar el Proyecto

```bash
cd days_15_21_llm/day21_small_transformer
cargo build --release
```

### Comandos Disponibles

#### 1. Demostración Completa

```bash
cargo run --release -- demo
```

Ejecuta una demostración completa que:
1. Construye el vocabulario desde el corpus
2. Crea el modelo transformer
3. Prepara el dataset
4. Simula entrenamiento (forward pass + métricas)
5. Genera texto desde múltiples prompts

#### 2. Información del Modelo

```bash
cargo run --release -- info
```

Muestra la arquitectura y configuración del modelo.

#### 3. Generar Texto

```bash
cargo run --release -- generate "Abeni was"
cargo run --release -- generate "The marketplace"
cargo run --release -- generate "Children sang"
```

Genera texto de forma autoregressiva desde un prompt dado.

#### 4. Entrenar Modelo

```bash
cargo run --release -- train
```

Ejecuta el loop de entrenamiento (50 épocas) y guarda la configuración.

## 📂 Estructura del Proyecto

```
src/
├── main.rs              # CLI y funciones principales
├── attention.rs         # Multi-Head Attention
├── positional.rs        # Positional Encoding
├── feedforward.rs       # Feed-Forward Networks
├── layer_norm.rs        # Layer Normalization
├── embedding.rs         # Embedding Layer
├── encoder.rs           # Encoder (no usado en LM puro)
├── decoder.rs           # Decoder Stack
├── transformer.rs       # Arquitectura completa
├── tokenizer.rs         # Tokenizer simple
├── dataset.rs           # Dataset loader
├── training.rs          # Loop de entrenamiento
├── generation.rs        # Generación autoregressiva
├── checkpoint.rs        # Guardar/cargar configuración
└── utils.rs             # Utilidades (softmax, loss, etc.)
```

## 🧮 Ecuaciones Clave Implementadas

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

### Positional Encoding

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Layer Normalization

```
LayerNorm(x) = γ * (x - μ) / (σ + ε) + β
```

### Feed-Forward Network

```
FFN(x) = max(0, xW1 + b1)W2 + b2
```

## 📊 Dataset

El proyecto usa un extracto de **Africa Galore** integrado en `dataset.rs`, que incluye:
- 10 párrafos de ejemplo
- Temas: arte, historia, deporte, cultura, naturaleza
- ~200 palabras únicas
- Vocabulario final: ~30 tokens (con filtrado de min_freq=2)

## ⚙️ Detalles de Implementación

### ¿Qué Está Completamente Implementado?

✅ **Forward Pass Completo**:
- Embeddings → Positional Encoding → Decoder → Output Projection
- Multi-Head Attention con máscaras causales
- Feed-Forward Networks con ReLU
- Layer Normalization
- Residual Connections

✅ **Generación Autoregressiva**:
- Sampling con temperatura
- Predicción greedy
- Manejo de secuencias de longitud variable

✅ **Métricas**:
- Cross-Entropy Loss
- Perplexity

### ⚠️ Limitaciones (Implementación Educativa)

Este es un proyecto **educativo simplificado**. Para un transformer completo en producción se requeriría:

1. **Backpropagation Completa** (~1500+ líneas adicionales)
   - Chain rule para todas las capas
   - Gradientes para attention, feedforward, layer norm
   - Actualización de pesos

2. **Optimizador Adam** (~300+ líneas)
   - First moment estimation
   - Second moment estimation
   - Bias correction

3. **Características Adicionales**:
   - Learning rate scheduling
   - Gradient clipping
   - Dropout
   - Weight decay
   - Batching eficiente
   - Mixed precision training

**Total estimado**: ~3000+ líneas de código adicional para entrenamiento real.

## 🎯 Conceptos Demostrados

Este proyecto demuestra exitosamente:

1. ✅ **Arquitectura Transformer completa** (decoder-only)
2. ✅ **Self-Attention** con múltiples cabezas
3. ✅ **Positional Information** mediante encoding sinusoidal
4. ✅ **Máscaras Causales** para prevenir ver el futuro
5. ✅ **Normalización y Residuales** para entrenamiento estable
6. ✅ **Generación Autoregressiva** token por token
7. ✅ **Pipeline Completo** desde texto → tokens → modelo → texto

## 📚 Aprendizajes del Plan de 21 Días

### Días 1-14: Fundamentos de Redes Neuronales
- Neuronas, activaciones, backpropagation
- Optimizadores (SGD, Momentum, Adam)
- Regularización (L1, L2, Dropout)
- CNNs para imágenes (MNIST)

### Días 15-21: LLMs y Transformers
- **Día 15-16**: N-grams y métricas de lenguaje
- **Día 17**: Tokenización con BPE
- **Día 18**: Word Embeddings (Skip-gram)
- **Día 19**: MLPs para clasificación de texto
- **Día 20**: Mecanismo de Atención
- **Día 21**: **Transformer Completo** ✨

## 🔍 Testing

Ejecutar tests:

```bash
cargo test
```

## 📖 Referencias

Este proyecto transpila y adapta los conceptos de los notebooks de Google DeepMind AI Foundations:
- Lab 4.1: Attention Visualization
- Lab 4.2: Implement Attention Equation
- Lab 4.3: Masked Multi-Head Attention
- Lab 4.4: Positional Embeddings
- Lab 1.5: Train Your Own Small Language Model
- Lab 2.6: Train SLM with BPE Tokenizer

## 🎓 Para Estudiantes

Este proyecto es ideal para:
- Entender **cómo funciona un transformer** internamente
- Aprender **implementación desde cero** sin abstracciones mágicas
- Ver **todas las piezas del puzzle** en un solo lugar
- Experimentar con arquitecturas pequeñas y rápidas

## 🙏 Agradecimientos

Proyecto educativo basado en:
- Google DeepMind AI Foundations Course
- "Attention Is All You Need" (Vaswani et al., 2017)
- The Illustrated Transformer (Jay Alammar)

---

**🎉 FELICITACIONES: Has completado el plan de 21 días de aprendizaje de LLMs desde cero!**
