# 🧠 MLP para Clasificación de Texto

**Día 19 del Desafío de 21 Días** - Redes Neuronales para NLP

## Descripción del Proyecto

Implementación completa de un clasificador de texto usando **Multi-Layer Perceptron (MLP)** con capa de embeddings en Rust. Este proyecto demuestra cómo combinar embeddings de palabras con redes neuronales densas para realizar análisis de sentimientos.

El proyecto incluye una implementación desde cero de:
- **Capa de Embeddings**: Representación vectorial de palabras
- **MLP multi-capa**: Red neuronal con capas ocultas y activación ReLU
- **Cross-Entropy Loss**: Función de pérdida para clasificación multi-clase
- **Backpropagation**: Entrenamiento end-to-end de embeddings + MLP
- **Batch Processing**: Procesamiento eficiente por lotes

## Características Implementadas

- ✅ MLP para clasificación de texto
- ✅ Capa de embedding trainable
- ✅ Batch processing con DataLoader
- ✅ Cross-entropy loss para clasificación multi-clase
- ✅ Tokenizador simple basado en palabras
- ✅ Métricas de evaluación (accuracy)
- ✅ Dataset sintético de análisis de sentimientos (positivo/negativo/neutral)
- ✅ Backpropagation completa para embeddings y MLP

## Arquitectura del Modelo

```
Texto → Tokenizador → Embedding Layer → MLP → Softmax → Clase
        (palabras)    (promedio)       (ReLU)  (probs)
```

**Configuración por defecto:**
- Embedding dimension: 32
- Hidden layers: [64, 32]
- Clases: 3 (positivo, negativo, neutral)
- Optimizer: SGD
- Learning rate: 0.01
- Batch size: 8
- Epochs: 50

## Estructura del Código

```
day19_mlp_text/
├── src/
│   ├── main.rs              # Punto de entrada con ejemplo de entrenamiento
│   ├── mlp.rs               # Implementación de MLP con backprop
│   ├── text_classifier.rs   # Clasificador completo (Embedding + MLP)
│   └── batch.rs             # DataLoader, Batch, Tokenizer
├── Cargo.toml
└── README.md
```

### Módulos Principales

#### `mlp.rs`
- Struct `Layer`: Capa individual con pesos, biases y gradientes
- Struct `MLP`: Red multi-capa con forward/backward pass
- Funciones: `relu()`, `relu_derivative()`, `softmax()`, `sigmoid()`

#### `text_classifier.rs`
- Struct `EmbeddingLayer`: Capa de embeddings trainable
- Struct `TextClassifier`: Modelo completo embedding + MLP
- Funciones: `cross_entropy_loss()`, `cross_entropy_gradient()`

#### `batch.rs`
- Struct `Batch`: Lote de datos para entrenamiento
- Struct `DataLoader`: Iterador sobre batches con shuffle
- Struct `SimpleTokenizer`: Tokenizador basado en palabras
- Función: `average_embeddings()` - Promedia embeddings de tokens

## Cómo Ejecutar

```bash
# Compilar y ejecutar
cd days_15_21_llm/day19_mlp_text
cargo run --release

# Ejecutar tests
cargo test

# Ver documentación
cargo doc --open
```

## Ejemplo de Salida

```
🧠 MLP Text - Día 19: Clasificación de Texto con MLP
================================================

📊 Creando dataset de sentimientos...
   - 36 ejemplos de entrenamiento
   - 6 ejemplos de prueba
   - Clases: ["positivo", "negativo", "neutral"]

📝 Construyendo vocabulario...
   - Vocabulario: 87 palabras

🏗️  Creando modelo...
   - Embedding dim: 32
   - Hidden layers: [64, 32]
   - Clases: 3

🎓 Entrenando modelo...

Epoch   0 | Loss: 1.0986 | Train Acc: 33.33% | Test Acc: 33.33%
Epoch  10 | Loss: 0.7234 | Train Acc: 75.00% | Test Acc: 66.67%
Epoch  20 | Loss: 0.4521 | Train Acc: 88.89% | Test Acc: 83.33%
Epoch  30 | Loss: 0.2890 | Train Acc: 94.44% | Test Acc: 100.00%
Epoch  40 | Loss: 0.1876 | Train Acc: 97.22% | Test Acc: 100.00%
Epoch  49 | Loss: 0.1298 | Train Acc: 100.00% | Test Acc: 100.00%

✅ Entrenamiento completado!

🔮 Probando predicciones:

   "me encanta este producto es increíble"
   → Clase predicha: 0 (positivo)

   "muy malo no lo recomiendo"
   → Clase predicha: 1 (negativo)

   "está bien nada especial"
   → Clase predicha: 2 (neutral)
```

## Conceptos Teóricos

### Embedding Layer
Una matriz de lookup que convierte IDs de tokens en vectores densos de dimensión fija. Los embeddings se aprenden durante el entrenamiento mediante backpropagation.

**Forward pass:**
```rust
embedding_avg = promedio(embeddings[token_ids])
```

**Backward pass:**
```rust
grad_embeddings[token_id] += grad_output / num_tokens
```

### Multi-Layer Perceptron (MLP)
Red neuronal feedforward con capas totalmente conectadas. Cada capa aplica:
```
z = input · W + b
output = ReLU(z)  // en capas ocultas
```

### Cross-Entropy Loss
Función de pérdida para clasificación multi-clase:
```
L = -log(p_target)
```

Donde `p_target` es la probabilidad predicha para la clase correcta después de softmax.

**Gradiente simplificado (con softmax):**
```
grad = predictions - one_hot(targets)
```

### Backpropagation
Algoritmo para calcular gradientes y actualizar pesos:
1. Forward pass: calcular predicciones
2. Calcular pérdida
3. Backward pass: propagar gradientes desde salida hacia entrada
4. Actualizar pesos: `W = W - learning_rate × grad_W`

## Dataset

El proyecto incluye un dataset sintético de análisis de sentimientos en español con:
- **36 ejemplos de entrenamiento** (12 por clase)
- **6 ejemplos de prueba** (2 por clase)
- **3 clases**: positivo, negativo, neutral

El dataset está diseñado para demostrar el funcionamiento del clasificador y puede ser reemplazado fácilmente por datos reales.

## Extensiones Posibles

- [ ] Implementar optimizador Adam en lugar de SGD
- [ ] Agregar regularización L2
- [ ] Visualizar curvas de aprendizaje con `plotters`
- [ ] Implementar dropout para prevenir overfitting
- [ ] Usar embeddings pre-entrenados (Word2Vec, GloVe)
- [ ] Concatenar embeddings en lugar de promediarlos
- [ ] Añadir capa de atención antes del MLP
- [ ] Entrenar en datasets reales (IMDB, Yelp)

## Referencias

Basado en los conceptos de:
- Lab 3.1-3.4: Redes neuronales y MLP (Notebooks de Google DeepMind)
- Técnicas de embedding para NLP
- Clasificación de texto con deep learning

---

