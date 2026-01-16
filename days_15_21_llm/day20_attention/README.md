# ⚡ Mecanismo de Atención

**Día 20 del Desafío de 21 Días** - Mecanismo de Atención

## Descripción del Proyecto

Implementación completa del mecanismo de atención, el componente fundamental de la arquitectura Transformer. Este proyecto implementa desde cero todos los elementos clave que permiten a los modelos de lenguaje "atender" a diferentes partes de una secuencia de entrada.

## Características Implementadas

- ✅ **Scaled Dot-Product Attention**: Implementación de la fórmula básica de atención
  - `Attention(Q,K,V) = softmax(QK^T / √d_k)V`
  - Función softmax estable numéricamente
  - Soporte para máscaras de atención

- ✅ **Máscaras de Atención**:
  - Máscara causal (para decoders autorregresivos)
  - Máscara de padding (para secuencias de longitud variable)

- ✅ **Atención con Parámetros Entrenables**:
  - Proyecciones Q, K, V
  - Proyección de salida
  - Inicialización Xavier/Glorot

- ✅ **Multi-Head Attention**: Atención con múltiples cabezas
  - Permite atender a diferentes subespacios de representación
  - Concatenación y proyección final
  - Soporte para encoder-decoder attention

- ✅ **Positional Encoding**: Codificación de posición
  - Codificación sinusoidal (fija)
  - Codificación aprendible (entrenable)
  - Fórmulas del paper "Attention is All You Need"

- ✅ **Visualización**: Heatmaps de atención con plotters
  - Visualización de pesos de atención individuales
  - Visualización de todas las cabezas de multi-head attention
  - Visualización de positional encoding
  - Visualización con etiquetas de tokens

## Estructura del Proyecto

```
day20_attention/
├── src/
│   ├── main.rs              # 7 ejemplos demostrativos completos
│   ├── attention.rs         # Scaled Dot-Product Attention y máscaras
│   ├── multihead.rs         # Multi-Head Attention
│   ├── positional.rs        # Positional Encoding sinusoidal y aprendible
│   ├── visualize.rs         # Heatmaps y visualizaciones
│   ├── order_invariance.rs  # Demostración de invariancia de orden
│   └── params.rs            # Conteo de parámetros entrenables
├── Cargo.toml
└── README.md
```

## Cómo Ejecutar

```bash
cd days_15_21_llm/day20_attention

# Compilar y ejecutar
cargo run --release

# Ejecutar tests
cargo test --release
```

## Ejemplos Demostrativos

El programa incluye 5 ejemplos educativos:

### 1. Scaled Dot-Product Attention Básico
Demuestra el mecanismo fundamental de atención con queries, keys y values simples.

```rust
Attention(Q,K,V) = softmax(QK^T / √d_k)V
```

**Salida**: `attention_basic.png`

### 2. Atención con Máscara Causal
Muestra cómo funciona la máscara causal que previene atender a tokens futuros (esencial en decoders).

```
Máscara causal (5x5):
  ✓   X   X   X   X    <- Posición 0 solo puede atender a sí misma
  ✓   ✓   X   X   X    <- Posición 1 puede atender a 0 y 1
  ✓   ✓   ✓   X   X    <- Y así sucesivamente...
  ✓   ✓   ✓   ✓   X
  ✓   ✓   ✓   ✓   ✓
```

**Salida**: `attention_causal.png`

### 3. Multi-Head Attention
Implementación de 8 cabezas de atención paralelas, cada una aprendiendo diferentes patrones.

```
Configuración:
- d_model: 64
- num_heads: 8
- d_k por cabeza: 8
```

**Salida**: `multihead_attention.png` (grid de 8 heatmaps)

### 4. Positional Encoding
Codificación posicional sinusoidal para inyectar información de posición en los embeddings.

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Salida**: `positional_encoding.png`

### 5. Pipeline Completo
Integración de embeddings + positional encoding + self-attention simulando el procesamiento de una frase.

```
Secuencia: "El gato come pescado ahora"
Pipeline: Embeddings → Positional Encoding → Self-Attention
```

**Salida**: `attention_with_tokens.png`

## Conceptos Teóricos

### ¿Qué es Atención?

El mecanismo de atención permite que cada elemento de una secuencia "atienda" a todos los demás elementos, aprendiendo qué partes son relevantes para procesar cada posición.

**Componentes**:
- **Q (Queries)**: "¿Qué estoy buscando?"
- **K (Keys)**: "¿Qué tengo para ofrecer?"
- **V (Values)**: "¿Qué información contiene?"

### Scaled Dot-Product Attention

La atención se calcula en tres pasos:

1. **Similitud**: `scores = Q · K^T` (producto punto)
2. **Escalado**: `scores / √d_k` (previene gradientes pequeños)
3. **Ponderación**: `softmax(scores) · V` (combina valores)

### Multi-Head Attention

En lugar de una sola atención, usa múltiples cabezas en paralelo:

```
MultiHead(Q,K,V) = Concat(head₁, ..., headₕ)W^O

donde head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

**Ventajas**:
- Cada cabeza aprende diferentes tipos de relaciones
- Aumenta la capacidad del modelo sin aumentar mucho el costo
- Permite capturar patrones sintácticos y semánticos diversos

### Positional Encoding

Los transformers no tienen noción inherente de orden secuencial. El positional encoding añade información de posición:

- **Sinusoidal (fijo)**: No requiere entrenamiento, generaliza bien
- **Aprendible**: Puede adaptarse mejor al dataset específico

## Arquitectura del Código

### attention.rs
```rust
// Función principal
pub fn scaled_dot_product_attention(
    queries: &Array2<f32>,
    keys: &Array2<f32>,
    values: &Array2<f32>,
    mask: Option<&Array2<f32>>,
) -> (Array2<f32>, Array2<f32>)

// Estructura con parámetros entrenables
pub struct Attention {
    pub w_q: Array2<f32>,  // Proyección Q
    pub w_k: Array2<f32>,  // Proyección K
    pub w_v: Array2<f32>,  // Proyección V
    pub w_o: Array2<f32>,  // Proyección salida
}
```

### multihead.rs
```rust
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub w_q: Vec<Array2<f32>>,  // Una proyección Q por cabeza
    pub w_k: Vec<Array2<f32>>,  // Una proyección K por cabeza
    pub w_v: Vec<Array2<f32>>,  // Una proyección V por cabeza
    pub w_o: Array2<f32>,       // Proyección final compartida
}
```

### positional.rs
```rust
pub struct PositionalEncoding {
    encoding: Array2<f32>,
    encoding_type: EncodingType,  // Sinusoidal o Learnable
}

// Aplicar a embeddings
pub fn apply(&self, embeddings: &Array2<f32>) -> Array2<f32>
```

### visualize.rs
```rust
// Visualizar atención simple
pub fn plot_attention_heatmap(...)

// Visualizar multi-head
pub fn plot_multihead_attention(...)

// Visualizar positional encoding
pub fn plot_positional_encoding(...)

// Visualizar con tokens
pub fn plot_attention_with_tokens(...)
```

## Tests

El proyecto incluye 14 tests unitarios:

```bash
cargo test --release

test result: ok. 14 passed; 0 failed; 0 ignored
```

**Tests incluidos**:
- Softmax suma 1 y está en [0,1]
- Scaled dot-product attention tiene dimensiones correctas
- Máscara causal bloquea posiciones futuras
- Multi-head attention con dimensiones inválidas falla
- Concatenación de cabezas funciona correctamente
- Positional encoding está en rango [-1,1]
- Aplicación de positional encoding mantiene dimensiones
- Y más...

## Relación con Transformers

Este proyecto implementa los bloques fundamentales usados en modelos como:
- **GPT** (Generative Pre-trained Transformer)
- **BERT** (Bidirectional Encoder Representations from Transformers)
- **T5** (Text-to-Text Transfer Transformer)

En el **Día 21** integraremos estos componentes en un Transformer completo.

## Dependencias

```toml
[dependencies]
ndarray = "0.15"        # Arrays N-dimensionales
ndarray-rand = "0.14"   # Inicialización aleatoria
rand = "0.8"            # Números aleatorios
plotters = "0.3"        # Visualización de heatmaps
```

## Siguientes Pasos

Este proyecto prepara el camino para:
- **Feed-Forward Networks** (capas FFN en transformers)
- **Layer Normalization** (normalización de capas)
- **Residual Connections** (conexiones residuales)
- **Transformer Encoder y Decoder** (arquitectura completa)
- **Small Language Model** (Día 21)

## Referencias

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Paper original de Transformers
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Guía visual
- [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) - Implementación anotada

---

**Nota**: Este proyecto es parte del plan maestro de aprendizaje de LLMs (días 15-21).
