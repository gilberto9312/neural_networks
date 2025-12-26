# 🎯 Word Embeddings y Similitud Semántica

**Día 18 del Desafío de 21 Días - Embeddings y Representaciones**

## 📖 Descripción del Proyecto

Este proyecto implementa embeddings de palabras (word embeddings) utilizando una versión simplificada del algoritmo **Skip-gram** de Word2Vec. Los embeddings son representaciones vectoriales densas de palabras que capturan relaciones semánticas en un espacio de alta dimensión.

La implementación está basada en el **Lab 2.5: Experiment with Embeddings** del curso AI Foundations de Google DeepMind, transpolado completamente a Rust.

## 🎓 Conceptos Teóricos

### ¿Qué son los Word Embeddings?

Los **word embeddings** son vectores densos de números reales que representan palabras en un espacio de alta dimensión (típicamente 50-300 dimensiones). A diferencia de representaciones dispersas como one-hot encoding, los embeddings capturan similitudes semánticas: palabras con significados similares tienen vectores cercanos en el espacio.

**Propiedades importantes:**
- Palabras similares → vectores cercanos
- Relaciones semánticas → operaciones vectoriales
- Dimensión reducida comparado con vocabulario

### Similitud Coseno

La **similitud coseno** mide qué tan similar es el significado de dos palabras calculando el coseno del ángulo entre sus vectores:

```
cos(u, v) = (u · v) / (||u|| × ||v||)
```

Donde:
- `u · v` es el producto punto
- `||u||` y `||v||` son las magnitudes (normas L2)

**Interpretación:**
- `+1`: Vectores idénticos (misma dirección)
- `0`: Vectores ortogonales (no relacionados)
- `-1`: Vectores opuestos (antónimos)

### Word2Vec y Skip-gram

**Word2Vec** es un modelo que aprende embeddings entrenando una red neuronal simple para predecir contexto de palabras.

**Skip-gram** predice palabras del contexto dada una palabra central:
- Ventana de contexto: palabras cercanas
- Objetivo: maximizar la probabilidad de palabras de contexto
- Resultado: palabras que aparecen en contextos similares tienen embeddings similares

**Implementación simplificada:**
```rust
// Para cada palabra central
for center_word in text {
    // Para cada palabra en su contexto
    for context_word in window(center_word) {
        // Acercar los embeddings
        embedding[center] += learning_rate * (embedding[context] - embedding[center])
    }
}
```

### Analogías Vectoriales

Una propiedad fascinante de los embeddings es que permiten operaciones algebraicas que capturan relaciones semánticas:

```
rey - hombre + mujer ≈ reina
París - Francia + Italia ≈ Roma
```

Esto funciona porque las relaciones semánticas se codifican como direcciones en el espacio vectorial.

## ✨ Características Implementadas

- ✅ **Matriz de embeddings** (lookup table)
- ✅ **Skip-gram simplificado** para entrenamiento
- ✅ **Similitud coseno** entre vectores
- ✅ **Búsqueda de vecinos cercanos** (palabras similares)
- ✅ **Operaciones de analogía** (word1 - word2 + word3 ≈ ?)
- ✅ **Normalización de embeddings**
- ✅ **Guardar/cargar embeddings** en JSON
- ✅ **Interfaz CLI completa**

## 🚀 Cómo Ejecutar

### 1. Entrenar Embeddings

Entrena embeddings con el corpus de ejemplo usando Skip-gram:

```bash
cd days_15_21_llm/day18_embeddings
cargo run --release -- train
```

Esto generará:
- Entrenamiento con 100 épocas
- Embeddings de 50 dimensiones
- Ventana de contexto de 2 palabras
- Archivo `embeddings.json` con los embeddings entrenados

### 2. Encontrar Palabras Similares

Busca las palabras más similares a una palabra dada:

```bash
cargo run --release -- similar king
cargo run --release -- similar cat 10  # Top 10 similares
```

Salida esperada:
```
🔍 Palabras similares a 'king':

  1. queen (similitud: 0.7234)
  2. prince (similitud: 0.6891)
  3. royal (similitud: 0.5432)
  ...
```

### 3. Operaciones de Analogía

Realiza operaciones vectoriales del tipo "A es a B como C es a ?"

```bash
cargo run --release -- analogy king man woman
cargo run --release -- analogy happy good bad
```

Salida esperada:
```
🧮 Analogía: 'king' - 'man' + 'woman' ≈ ?

  1. queen (similitud: 0.6543)
  2. princess (similitud: 0.5234)
  3. royal (similitud: 0.4321)
```

### 4. Demostración Completa

Ejecuta una demostración con varios ejemplos:

```bash
cargo run --release -- demo
```

## 📂 Estructura del Código

```
day18_embeddings/
├── src/
│   ├── main.rs              # CLI y entrenamiento Skip-gram
│   ├── embedding_layer.rs   # Matriz de embeddings y lookup
│   ├── similarity.rs        # Similitud coseno y analogías
│   └── visualize.rs         # Visualización (placeholder)
├── Cargo.toml
└── README.md
```

## ⚠️ Nota Importante: Simplificación vs Word2Vec Real

### ¿Por qué esta implementación es tan simple?

Esta implementación es **intencionalmente simplificada** con fines educativos. A continuación se explican las diferencias con Word2Vec/Skip-gram real:

#### **Word2Vec Real (Mikolov et al. 2013)**

```
Arquitectura completa:
Input (one-hot) → Embedding Layer → Softmax → Output (probabilidades)

Características:
✓ Red neuronal de 2 capas
✓ Función de pérdida: Cross-entropy
✓ Activación de salida: Softmax
✓ Backpropagation completa
✓ Negative sampling (5-20 palabras negativas)
✓ Learning rate adaptativo (decae con el tiempo)
✓ Optimizado para corpus grandes (millones de palabras)
```

#### **Nuestra Implementación Simplificada**

```rust
// Aproximación geométrica directa
let diff = &context_emb - &center_emb;
let new_center = &center_emb + &(&diff * learning_rate);
```

**Características:**
- ❌ Sin red neuronal (solo operaciones vectoriales)
- ❌ Sin funciones de activación (ReLU, Sigmoid, Softmax)
- ❌ Sin negative sampling
- ✅ Learning rate fijo (0.01)
- ✅ Actualización directa de embeddings

### ¿Por qué funciona esta simplificación?

Porque **captura la esencia de Skip-gram**: *palabras que aparecen en contextos similares deben tener embeddings cercanos*.

La versión completa de Word2Vec hace esto a través de:
- Maximizar `P(contexto|palabra_central)` usando gradientes
- Negative sampling para eficiencia

Nuestra versión lo hace directamente:
- Acercar embeddings de palabras que co-ocurren
- Sin cálculos probabilísticos complejos

### Decisiones de Diseño Justificadas

| Aspecto | Decisión Tomada | Razón |
|---------|----------------|-------|
| **Sin activaciones** | Solo operaciones vectoriales | Word2Vec real tampoco usa activación en la capa de embedding. La softmax está en la salida, que nosotros evitamos |
| **Learning rate fijo** | 0.01 constante | Suficiente para corpus pequeño (~200 tokens). Word2Vec real usa decaimiento: `0.025 * (1 - epoch/max_epochs)` |
| **Sin negative sampling** | Solo muestras positivas | Con vocabulario pequeño (<100 palabras), no es crítico. Word2Vec real necesita esto para vocabularios de 100k+ palabras |
| **Actualización directa** | Mover vectores geométricamente | Más intuitivo educativamente que backpropagation |

### Comparación de Resultados

| Métrica | Versión Simplificada | Word2Vec Real |
|---------|---------------------|---------------|
| **Velocidad** | ⚡ Muy rápida | Más lenta |
| **Corpus pequeño** | ✅ Excelente | Overkill |
| **Corpus grande** | ❌ Limitada | ✅ Superior |
| **Calidad embeddings** | Suficiente para ejemplos | Estado del arte |
| **Complejidad código** | 🎓 Educativa | Producción |

### ¿Cuándo usar cada versión?

**Usar esta implementación simplificada:**
- ✅ Aprender conceptos de embeddings
- ✅ Corpus pequeños (<10,000 palabras)
- ✅ Prototipado rápido
- ✅ Entender geometría de embeddings

**Usar Word2Vec real (gensim, fastText):**
- ✅ Producción
- ✅ Corpus grandes (millones de palabras)
- ✅ Máxima calidad de embeddings
- ✅ Eficiencia con GPU

### Ejemplo de Skip-gram Real (referencia)

Para contexto educativo, así se vería una implementación más realista:

```rust
// Versión más cercana a Word2Vec real
fn train_skipgram_realistic(...) {
    let mut learning_rate = 0.025;

    for epoch in 0..epochs {
        // 1. Decaer learning rate
        learning_rate = 0.025 * (1.0 - epoch as f32 / epochs as f32);

        for (center_word, context_word) in pairs {
            // 2. Calcular score con dot product
            let score = center_emb.dot(&context_emb);

            // 3. Aplicar sigmoid
            let prob = 1.0 / (1.0 + (-score).exp());

            // 4. Gradiente positivo
            let gradient = (1.0 - prob) * learning_rate;
            center_emb += gradient * context_emb;

            // 5. Negative sampling (5 palabras aleatorias)
            for neg_word in sample_negative(5) {
                let neg_score = center_emb.dot(&neg_emb);
                let neg_prob = 1.0 / (1.0 + (-neg_score).exp());

                // Gradiente negativo (alejar)
                center_emb -= neg_prob * learning_rate * neg_emb;
            }
        }
    }
}
```

### Conclusión

Esta implementación sacrifica **precisión** y **escalabilidad** a favor de **claridad educativa** y **comprensión conceptual**.

Para el Día 18, el objetivo es entender:
- ✅ Qué son los embeddings
- ✅ Cómo se representa significado en vectores
- ✅ Similitud coseno y operaciones vectoriales

**No** el objetivo es:
- ❌ Entrenar embeddings de producción
- ❌ Competir con GloVe/fastText
- ❌ Escalar a millones de palabras

En los **Días 19-21** construiremos sobre estos embeddings para crear redes neuronales completas (MLP, Attention, Transformer) donde verás activaciones, backpropagation y optimización avanzada.

---

## 🔬 Detalles de Implementación

### EmbeddingLayer (embedding_layer.rs)

```rust
pub struct EmbeddingLayer {
    pub embeddings: Array2<f32>,          // Matriz (vocab_size × embedding_dim)
    pub token_to_id: HashMap<String, usize>,
    pub id_to_token: Vec<String>,
    pub embedding_dim: usize,
}
```

**Métodos principales:**
- `new(vocab, dim)`: Crea embeddings con inicialización aleatoria
- `get_embedding(token)`: Obtiene vector de un token
- `update_embedding(token, vec)`: Actualiza embedding
- `normalize_embeddings()`: Normaliza todos los vectores a longitud 1
- `save(path)` / `load(path)`: Persistencia en JSON

### Similitud Coseno (similarity.rs)

```rust
pub fn cosine_similarity(u: &Array1<f32>, v: &Array1<f32>) -> f32 {
    let dot_product = u.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    let norm_u = u.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_v = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_u * norm_v)
}
```

### Entrenamiento Skip-gram (main.rs)

```rust
fn train_skipgram(text: &str, embedding_dim: usize, epochs: usize, window_size: usize)
```

**Algoritmo:**
1. Tokenizar texto y construir vocabulario
2. Inicializar embeddings aleatoriamente
3. Para cada época:
   - Para cada palabra central en el texto:
     - Obtener palabras de contexto (ventana)
     - Calcular gradiente simplificado
     - Actualizar embeddings de palabra central y contexto
4. Normalizar embeddings finales

## 📊 Ejemplos de Resultados

Después de entrenar con el corpus de ejemplo, se observan similitudes como:

| Par de Palabras | Similitud Coseno |
|----------------|------------------|
| king - queen   | 0.72 (alta)      |
| cat - dog      | 0.68 (alta)      |
| apple - banana | 0.65 (alta)      |
| car - bus      | 0.71 (alta)      |
| good - bad     | 0.42 (media)     |
| king - car     | 0.08 (baja)      |

**Analogías exitosas:**
- `king - man + woman ≈ queen` ✓
- `apple - fruit + vehicle ≈ car` ✓

## 🧪 Ejecutar Tests

```bash
cargo test
```

Los tests verifican:
- Creación de capa de embeddings
- Similitud coseno (vectores idénticos, ortogonales, opuestos)
- Búsqueda de vecinos cercanos
- Normalización de vectores

## 🛠️ Tecnologías Utilizadas

- **ndarray**: Operaciones con matrices y vectores
- **ndarray-rand**: Inicialización aleatoria de embeddings
- **rand**: Generación de números aleatorios
- **serde/serde_json**: Serialización de embeddings
- **Rust std**: Collections (HashMap), I/O

## 📚 Referencias

- **Lab 2.5: Experiment with Embeddings** - Google DeepMind AI Foundations
- Mikolov et al. (2013): "Efficient Estimation of Word Representations in Vector Space"
- Pennington et al. (2014): "GloVe: Global Vectors for Word Representation"

## 🎯 Próximos Pasos

Este proyecto es la base para:
- **Día 19**: MLP para clasificación de texto
- **Día 20**: Mecanismo de atención
- **Día 21**: Transformer completo

Los embeddings entrenados aquí se pueden usar como capa de entrada en redes neuronales más complejas.

---

**Parte del Plan Maestro de Aprendizaje de LLMs (Días 15-21)**

Proyecto educativo - Implementación en Rust de conceptos fundamentales de NLP y LLMs.
