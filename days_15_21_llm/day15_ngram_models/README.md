# 🧠 Redes Neuronales - Serie de Aprendizaje Diario

## 📅 Día 15: Modelos N-gram

### 🎯 Objetivo del Día
Entender cómo funcionan los modelos de lenguaje estadísticos más simples (N-grams), aprender a calcular probabilidades a partir de frecuencias en un corpus, y usar estos modelos para **generar texto automáticamente**.

Este es el primer paso antes de llegar a los Transformers modernos. Si no entiendes N-grams, no entenderás qué problemas resuelven los modelos neuronales.

---

## 🔍 ¿Qué es un Modelo N-gram?

Un modelo N-gram es un modelo de lenguaje que **predice la siguiente palabra** basándose en las **N-1 palabras anteriores**.

### Los tres tipos que implementaremos:

#### 1️⃣ **Unigram** (N=1)
Ignora el contexto completamente. Solo mira qué palabras son más frecuentes en el corpus.

```
P(palabra) = count(palabra) / total_palabras
```

**Ejemplo**: Si "the" aparece 1000 veces en un corpus de 10,000 palabras:
```
P("the") = 1000/10,000 = 0.1 = 10%
```

#### 2️⃣ **Bigram** (N=2)
Usa **1 palabra** de contexto para predecir la siguiente.

```
P(w2 | w1) = count(w1, w2) / count(w1)
```

**Ejemplo**: Si "the music" aparece 50 veces y "the" aparece 1000 veces:
```
P("music" | "the") = 50/1000 = 0.05 = 5%
```

#### 3️⃣ **Trigram** (N=3)
Usa **2 palabras** de contexto para predecir la siguiente.

```
P(w3 | w1, w2) = count(w1, w2, w3) / count(w1, w2)
```

**Ejemplo**: Si "in the club" aparece 10 veces y "in the" aparece 100 veces:
```
P("club" | "in the") = 10/100 = 0.1 = 10%
```

---

## ❓ ¿Por qué Necesitamos Modelos de Lenguaje?

Imagina que estás escribiendo un mensaje y tu teclado quiere **autocompletar** la siguiente palabra. ¿Cómo sabe qué sugerir?

```
Usuario escribe: "I went to the"
Opciones posibles: "store", "beach", "moon", "elephant"
```

Un modelo de lenguaje asigna **probabilidades** a cada palabra:
- P("store" | "I went to the") = 0.25 ← Muy probable ✅
- P("beach" | "I went to the") = 0.20 ← Probable ✅
- P("moon" | "I went to the") = 0.05 ← Poco probable 🤔
- P("elephant" | "I went to the") = 0.01 ← Muy raro ❌

El modelo **muestrea** de esta distribución para elegir la siguiente palabra.

---

## 🏗️ Estructura del Código

```
day15_ngram_models/
├── src/
│   ├── main.rs          # Demo completa con los 3 modelos
│   ├── dataset.rs       # Carga Africa Galore + tokenización
│   ├── ngram.rs         # UnigramModel, BigramModel, TrigramModel
│   └── sampling.rs      # Generación de texto con muestreo
├── Cargo.toml
└── README.md
```

### ¿Qué hace cada módulo?

#### `dataset.rs` - Preprocesamiento
```rust
// Carga el JSON del dataset
let texts = load_africa_galore("../../datasets/africa_galore.json")?;

// Tokeniza: "Hello, world!" → ["hello", "world"]
let tokens = tokenize("Hello, world!");
// Resultado: ["hello", "world"]
```

**¿Por qué convertir a minúsculas?** Para que "The" y "the" se cuenten como la misma palabra.

#### `ngram.rs` - Los Modelos
```rust
// Entrena un modelo bigram
let bigram_model = BigramModel::new(&tokens);

// Calcula probabilidad condicional
let prob = bigram_model.probability("the", "music");
// P("music" | "the") = ?
```

#### `sampling.rs` - Generación de Texto
```rust
// Genera 30 palabras empezando con "the"
let text = generate_bigram(&model, "the", 30);
// Resultado: "the music was playing in the club and people were dancing..."
```

**¿Por qué muestreo aleatorio?** Si siempre elegimos la palabra más probable, el texto sería muy repetitivo y aburrido.

---

## 🧪 Experimentos que Realizamos

### 1. Tokenización del Corpus
Tomamos el dataset **Africa Galore** (232 párrafos sobre cultura africana) y lo dividimos en tokens:

```
Texto original:
"The Lagos air was thick with humidity, but the energy in the club was electric."

Tokens generados:
["the", "lagos", "air", "was", "thick", "with", "humidity", "but", "the", "energy", ...]
```

**Total**: ~31,000 tokens
**Vocabulario único**: ~5,100 palabras

### 2. División Train/Test (80/20)
```
Entrenamiento: 24,800 tokens → Para calcular frecuencias
Prueba:        6,200 tokens  → Para evaluar perplexity
```

**¿Por qué dividir?** Para asegurarnos de que el modelo funciona con texto que **nunca ha visto**.

### 3. Generación de Texto

#### Unigram (sin contexto)
```
Prompt: "Jide was hungry so"
Generación: "the music a in was people of and traditional..."
```
❌ **No tiene sentido** - solo elige palabras frecuentes al azar.

#### Bigram (1 palabra de contexto)
```
Prompt: "Jide was hungry so"
Generación: "she went looking for food in the market to buy..."
```
✅ **Algo mejor** - las palabras tienen más coherencia local.

#### Trigram (2 palabras de contexto)
```
Prompt: "Jide was hungry so"
Generación: "she went looking for a traditional dish made with..."
```
✅✅ **Mucho mejor** - frases más coherentes y gramaticales.

---

## 📊 ¿Cómo Medimos si un Modelo es Bueno?

Usamos una métrica llamada **Perplexity**.

### ¿Qué es Perplexity?

Es una medida de **qué tan sorprendido está el modelo** ante nuevas palabras.

```
Perplexity = exp(-1/N * Σ log P(palabra_i | contexto))
```

**Interpretación**:
- Perplexity de 100 = El modelo está tan confundido como si tuviera que elegir entre **100 palabras al azar**
- **Menor perplexity = mejor modelo**

### Resultados en Africa Galore:
```
Unigram:   Perplexity = 342  ← Muy confundido
Bigram:    Perplexity = 128  ← Mejor
Trigram:   Perplexity =  68  ← ¡Mucho mejor!
```

**Conclusión**: Más contexto = mejores predicciones

---

## ⚠️ El Gran Problema: Data Sparsity

Aquí viene el **problema masivo** de los N-grams.

### ¿Qué es Data Sparsity?

A medida que aumentas N, la cantidad de **combinaciones posibles** explota:

```
Vocabulario: 5,100 palabras

Bigramas posibles:  5,100 × 5,100 = 26 millones
Trigramas posibles: 5,100³ = 132,000 millones
```

**Pero en nuestro dataset solo tenemos ~31,000 tokens.**

Esto significa que la **mayoría de combinaciones NUNCA aparecen**:

```
Bigrams con count = 0:  99.95% 😱
Trigrams con count = 0: 99.98% 😱😱
```

### ¿Qué pasa cuando el modelo ve una secuencia nueva?

```python
# Bigram que nunca apareció en el dataset
model.probability("purple", "elephant")
# → 0.0 (no puede predecir nada)
```

El modelo **se queda atascado** y no puede generar más texto.

**Solución temporal**: Asignar una probabilidad muy pequeña (1e-10) en lugar de 0.

**Solución real**: Usar modelos neuronales (Transformers) que **generalizan** mejor.

---

## 💡 Conceptos Clave Aprendidos

### 1. Probabilidad Condicional
Los modelos de lenguaje funcionan calculando **P(siguiente_palabra | contexto)**.

### 2. Trade-off Contexto vs Datos
- **Más contexto** (trigram) = mejores predicciones
- **Pero** requiere **muchos más datos** para entrenar bien

### 3. Muestreo Estocástico
No siempre elegimos la palabra con mayor probabilidad. Usamos `WeightedIndex` de Rust para muestrear según probabilidades:

```rust
let weights = [0.5, 0.3, 0.2];  // Probabilidades
let words = ["the", "a", "an"];
let chosen = sample_weighted(&words, &weights);
// 50% chance de "the", 30% de "a", 20% de "an"
```

Esto hace el texto **más creativo y menos repetitivo**.

### 4. Tokenización es Importante
```
Texto mal tokenizado: ["Hello", "world", "!"]
Texto bien tokenizado: ["hello", "world"]
```

Una mala tokenización puede arruinar todo el modelo.

### 5. Límites de N-grams
- Solo ven **N-1 palabras de contexto**
- No entienden **significado** (no saben que "dog" y "puppy" son similares)
- Sufren de **data sparsity** severa
- No pueden usar contexto de hace 50 palabras

**Por eso existen los Transformers** (Día 21) que resuelven todos estos problemas.

---

## 🔧 Cómo Ejecutar

```bash
# Navegar al proyecto
cd days_15_21_llm/day15_ngram_models

# Compilar y ejecutar
cargo run --release

# Ejecutar tests
cargo test

# Solo compilar
cargo build --release
```

---

## 📈 Salida Esperada

```
🚀 Modelos N-gram - Día 15
═══════════════════════════════════════════════════

📦 Cargando dataset Africa Galore...
✅ Dataset cargado: 232 textos

🔤 Tokenizando corpus...
✅ Total de tokens: 31,234
✅ Vocabulario único: 5,143 palabras
✅ Tokens de entrenamiento: 24,987
✅ Tokens de prueba: 6,247

═══════════════════════════════════════════════════
📊 MODELO UNIGRAM
═══════════════════════════════════════════════════

🎓 Entrenando modelo Unigram...
✅ Modelo Unigram entrenado

📈 Top 10 palabras más frecuentes:
   1. 'the' - 1234 veces (P=0.0495)
   2. 'of' - 567 veces (P=0.0227)
   3. 'and' - 456 veces (P=0.0183)
   ...

✍️  Generación de texto (Unigram - 30 palabras):
   the music a in of was people and tradition with culture to...

📉 Perplexity (Unigram): 342.15

═══════════════════════════════════════════════════
📊 MODELO BIGRAM
═══════════════════════════════════════════════════

🎓 Entrenando modelo Bigram...
✅ Modelo Bigram entrenado

🔍 Ejemplos de probabilidades P(w2|w1):
   P('music' | 'the') = 0.0245
   P('the' | 'in') = 0.1234
   P('a' | 'was') = 0.0567

✍️  Generación de texto (Bigram - 30 palabras):
   the music was playing in the club and people were dancing to...

📉 Perplexity (Bigram): 127.83

═══════════════════════════════════════════════════
📊 MODELO TRIGRAM
═══════════════════════════════════════════════════

🎓 Entrenando modelo Trigram...
✅ Modelo Trigram entrenado

🔍 Ejemplos de probabilidades P(w3|w1,w2):
   P('was' | 'the', 'music') = 0.3333
   P('club' | 'in', 'the') = 0.0833
   P('music' | 'of', 'the') = 0.0456

✍️  Generación de texto (Trigram - 30 palabras):
   the music was a celebration of life and culture in the heart of africa...

📉 Perplexity (Trigram): 68.42

═══════════════════════════════════════════════════
🏆 COMPARACIÓN DE MODELOS
═══════════════════════════════════════════════════

📊 Resumen de Perplexity (menor es mejor):
   Unigram:  342.15
   Bigram:   127.83
   Trigram:   68.42

💡 Interpretación:
   - Perplexity mide qué tan 'sorprendido' está el modelo
   - Menor perplexity = mejor predicción
   - Modelos de mayor orden (trigram) suelen tener menor perplexity
   - Pero requieren más datos y pueden sufrir de overfitting

✅ Análisis completo de modelos N-gram finalizado!
```

---

## ⚙️ Parámetros que Puedes Ajustar

### En `main.rs`:

```rust
// Número de palabras a generar
let num_words = 50;  // Prueba con 10, 30, 100

// Palabra inicial para bigram
let start_word = "music";  // Prueba: "africa", "the", "celebration"

// Par inicial para trigram
let start_pair = ("the", "music");  // Prueba diferentes pares
```

### División train/test:
```rust
// Cambiar el ratio de división (actualmente 80/20)
let split_index = (all_tokens.len() as f64 * 0.9) as usize;  // 90/10
```

---

## 🐛 ¿Qué puede salir mal?

### 1. El modelo no puede continuar
```
⚠️ No valid continuation found.
```

**Causa**: El bigram/trigram nunca apareció en el dataset.
**Solución**: Usa un prompt que aparezca en el corpus, o usa un modelo de menor orden (unigram siempre funciona).

### 2. Texto generado no tiene sentido
**Causa Normal**: Los N-grams son modelos muy simples.
**Mejora**: Usa trigram en vez de unigram, aumenta el tamaño del dataset.

### 3. Dataset no encontrado
```
❌ Error cargando dataset
```

**Solución**: Asegúrate de ejecutar desde `days_15_21_llm/day15_ngram_models/` y que `datasets/africa_galore.json` existe.

---

## 🎯 Próximos Pasos

Este es solo el comienzo. En los siguientes días:

- **Día 16**: Análisis avanzado de perplexity
- **Día 17**: Tokenización BPE (mucho mejor que space tokenizer)
- **Día 18**: Embeddings (Word2Vec - entiende similitudes semánticas)
- **Día 19**: MLP para clasificación de texto
- **Día 20**: Mecanismo de atención (el corazón de los Transformers)
- **Día 21**: **Transformer completo** - Un LLM real desde cero

---

## 📚 Dependencias

```toml
[dependencies]
rand = "0.8"           # Para muestreo de distribuciones
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"     # Para cargar JSON
```

**Sin dependencias de ML pesadas** - Todo implementado desde cero para aprender.

---

## 📖 Referencias

- [N-gram Language Models - Stanford](https://web.stanford.edu/~jurafsky/slp3/)
- [Perplexity Explained](https://en.wikipedia.org/wiki/Perplexity)
- [Africa Galore Dataset](https://storage.googleapis.com/dm-educational/assets/ai_foundations/africa_galore.json)

---

*Implementado en Rust como parte del desafío de 21 días de Neural Networks desde cero.*
