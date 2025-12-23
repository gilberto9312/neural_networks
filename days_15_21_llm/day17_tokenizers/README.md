# Tokenizadores desde Cero

**Día 17 del Desafío de 21 Días** - Tokenización y Preprocesamiento de Texto

## Descripción del Proyecto

Implementación completa de tres tipos de tokenizadores para preprocesar texto antes de entrenar modelos de lenguaje:

- **Tokenizador por Caracteres**: Divide el texto en caracteres individuales
- **Tokenizador por Palabras**: Divide el texto en palabras usando regex
- **Tokenizador BPE**: Byte Pair Encoding que aprende subpalabras del corpus

## Características Implementadas

- ✅ Tokenizador por caracteres (con soporte Unicode)
- ✅ Tokenizador por palabras (whitespace y regex)
- ✅ Tokenizador BPE (Byte Pair Encoding) completo
- ✅ Vocabulario y mapeo bidireccional token ↔ ID
- ✅ Padding y truncamiento de secuencias
- ✅ Tokens especiales (PAD, UNK, BOS, EOS, EOW)
- ✅ Serialización de vocabulario en JSON
- ✅ CLI para entrenar y codificar

## Instalación

```bash
cd days_15_21_llm/day17_tokenizers
cargo build --release
```

## Uso

### 1. Demo Interactiva

Ejecuta una demostración comparando los tres tokenizadores:

```bash
cargo run --release -- demo
```

### 2. Entrenar un Tokenizador BPE

Entrena un vocabulario BPE desde un corpus de texto:

```bash
cargo run --release -- train corpus.txt vocab.json 1000
```

Parámetros:
- `corpus.txt`: Archivo de texto con el corpus de entrenamiento
- `vocab.json`: Archivo donde se guardará el vocabulario
- `1000`: Número de merges a realizar (opcional, default: 500)

### 3. Codificar Texto

Codifica texto usando un vocabulario BPE previamente entrenado:

```bash
cargo run --release -- encode vocab.json "Hello world"
```

### 4. Comparar Tokenizadores

Compara cómo diferentes tokenizadores procesan varios textos:

```bash
cargo run --release -- compare
```

## Ejemplo con Africa Galore Dataset

### Preparar el corpus

```bash
# Extraer descripciones del dataset JSON
cd ../../datasets
cat africa_galore.json | jq -r '.[].description' > africa_corpus.txt
```

### Entrenar tokenizador BPE

```bash
cd ../days_15_21_llm/day17_tokenizers
cargo run --release -- train ../../datasets/africa_corpus.txt africa_bpe.json 2000
```

### Codificar texto

```bash
cargo run --release -- encode africa_bpe.json "The Lagos air was thick with humidity"
```

## Conceptos Teóricos

### Tokenización por Caracteres

**Ventajas:**
- Vocabulario muy pequeño (solo caracteres únicos)
- No hay problema de palabras fuera de vocabulario (OOV)
- Simple de implementar

**Desventajas:**
- Secuencias muy largas
- Dificulta el aprendizaje de patrones semánticos

### Tokenización por Palabras

**Ventajas:**
- Secuencias cortas
- Preserva unidades semánticas
- Intuitivo

**Desventajas:**
- Vocabulario muy grande
- Problema de palabras OOV
- Mal manejo de palabras raras o con errores

### Tokenización BPE (Byte Pair Encoding)

**Ventajas:**
- Balance entre tamaño de vocabulario y longitud de secuencia
- Maneja bien palabras OOV (divide en subpalabras)
- Captura morfología y patrones frecuentes

**Desventajas:**
- Más complejo de implementar
- Requiere entrenamiento previo
- Puede generar tokens sin significado lingüístico

#### Algoritmo BPE

1. **Inicialización**: Dividir corpus en caracteres + token especial `</w>`
2. **Conteo**: Contar frecuencia de todos los pares adyacentes
3. **Merge**: Fusionar el par más frecuente en un nuevo token
4. **Iteración**: Repetir pasos 2-3 hasta alcanzar el vocabulario deseado

**Ejemplo:**

```
Corpus inicial: ["desert", "deserted", "desert"]

Iteración 1:
  Palabras: [d e s e r t </w>], [d e s e r t e d </w>], [d e s e r t </w>]
  Par más frecuente: ('e', 's') → merge a 'es'

Iteración 2:
  Palabras: [d es e r t </w>], [d es e r t e d </w>], [d es e r t </w>]
  Par más frecuente: ('d', 'es') → merge a 'des'

...

Resultado final:
  Vocabulario: {d, e, s, r, t, </w>, es, des, dese, deser, desert, ...}
  Tokenización: "desert" → ["desert</w>"]
  Tokenización: "deserted" → ["desert", "ed</w>"]
```

## Estructura del Código

```
src/
├── main.rs              # CLI y comandos principales
├── vocab.rs             # Gestión de vocabulario y tokens especiales
├── char_tokenizer.rs    # Tokenizador por caracteres
├── word_tokenizer.rs    # Tokenizador por palabras
└── bpe_tokenizer.rs     # Implementación BPE completa
```

### Arquitectura

Todos los tokenizadores implementan la siguiente interfaz:

```rust
pub trait Tokenizer {
    fn tokenize(&self, text: &str) -> Vec<String>;
    fn encode(&self, text: &str) -> Vec<usize>;
    fn decode(&self, ids: &[usize]) -> String;
}
```

### Vocabulario

El módulo `vocab.rs` proporciona:

```rust
pub struct Vocabulary {
    token_to_id: HashMap<String, usize>,
    id_to_token: Vec<String>,
    special_tokens: SpecialTokens,
}
```

Tokens especiales:
- `<PAD>`: Padding para secuencias
- `<UNK>`: Tokens desconocidos
- `<BOS>`: Inicio de secuencia
- `<EOS>`: Fin de secuencia
- `</w>`: Fin de palabra (para BPE)

## Resultados de Ejemplo

### Comparación en "The music was electric"

| Tokenizador | Tokens | Vocab Size | Ejemplo de Tokens |
|------------|--------|------------|-------------------|
| Caracteres | 23 | ~30 | ['T', 'h', 'e', ' ', 'm', 'u', 's', 'i', 'c', ...] |
| Palabras | 4 | ~50 | ["the", "music", "was", "electric"] |
| BPE (50 merges) | 8-12 | ~70 | ["the</w>", "mu", "sic</w>", "was</w>", "ele", "c", "tri", "c</w>"] |

### Visualización de Vocabulario BPE

Los primeros tokens son caracteres individuales, luego subpalabras frecuentes:

```
0-4:   Tokens especiales (<PAD>, <UNK>, <BOS>, <EOS>, </w>)
5-36:  Caracteres (a, b, c, ..., z, espacio, puntuación)
37+:   Subpalabras aprendidas (th, he, the, ing, ed, ...</w>, etc.)
```

## Testing

Ejecutar tests unitarios:

```bash
cargo test
```

Ejecutar tests con output:

```bash
cargo test -- --nocapture
```

## Optimizaciones Futuras

- [ ] Caché de tokenización para textos repetidos
- [ ] Paralelización del conteo de pares (rayon)
- [ ] Soporte para múltiples idiomas
- [ ] Implementar WordPiece y Unigram tokenizers
- [ ] Visualización de distribución de tokens
- [ ] Benchmark de velocidad

## Referencias

**Labs de Google DeepMind:**
- Lab 2.1: Preprocesar datos
- Lab 2.2: Tokenización por caracteres y palabras
- Lab 2.3: Tokenización en subwords
- Lab 2.4: Implementar BPE tokenizer

**Papers:**
- Sennrich et al. (2016): "Neural Machine Translation of Rare Words with Subword Units"
- Gage (1994): "A New Algorithm for Data Compression"

## Próximos Pasos

Este tokenizador será utilizado en los siguientes días:

- **Día 18**: Embeddings - Convertir tokens en vectores densos
- **Día 19**: MLP para texto - Clasificación con embeddings
- **Día 21**: Small Transformer - Modelo de lenguaje completo

---

**Autor**: Proyecto educativo - 21 Días de Neural Networks en Rust
**Día**: 17/21
**Tema**: Tokenización y Preprocesamiento
