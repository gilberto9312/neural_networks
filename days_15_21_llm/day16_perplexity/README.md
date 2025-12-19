# 🧠 Redes Neuronales - Serie de Aprendizaje LLM

## 📅 Día 16: Evaluación con Perplexity

### 🎯 Objetivo del Día
Comprender cómo evaluar modelos de lenguaje usando perplexity, comparar el rendimiento de diferentes modelos N-gram, y entender por qué esta métrica es fundamental para medir la calidad de predicción en modelos de lenguaje.

### 🔍 ¿Qué es Perplexity?

Perplexity es una métrica que mide **qué tan "sorprendido" está un modelo** ante datos nuevos. Es la forma estándar de evaluar modelos de lenguaje.

#### Fórmula matemática:
```
Perplexity = exp(-1/N * Σ log P(palabra_i | contexto))
```

Donde:
- `N` es el número total de palabras
- `P(palabra_i | contexto)` es la probabilidad que el modelo asigna a cada palabra dado su contexto

#### Propiedades clave:
- **Menor perplexity = mejor modelo**: Un modelo con perplexity de 50 es mejor que uno con 100
- **Interpretación intuitiva**: Una perplexity de 100 significa que el modelo está tan confundido como si tuviera que elegir uniformemente entre 100 palabras
- **Sensibilidad al contexto**: Modelos de mayor orden (trigram) generalmente tienen menor perplexity que modelos simples (unigram)
- **Dependiente del dataset**: La perplexity solo es comparable entre modelos evaluados en el mismo conjunto de prueba

### 🏗️ Estructura del Código

Este proyecto extiende el trabajo del **Día 15** añadiendo análisis comparativo de perplexity.

#### Módulos utilizados (del Día 15):
1. **`ngram.rs`**: Modelos Unigram, Bigram, Trigram
2. **`dataset.rs`**: Carga de Africa Galore
3. **`sampling.rs`**: Generación de texto

#### Funciones de evaluación:
- `calculate_perplexity_unigram()`: Evalúa modelo unigram
- `calculate_perplexity_bigram()`: Evalúa modelo bigram
- `calculate_perplexity_trigram()`: Evalúa modelo trigram

### 🧪 Experimentos Realizados

#### 1. División Train/Test
- **80% entrenamiento**: Para estimar probabilidades
- **20% prueba**: Para evaluar perplexity sin sesgo

#### 2. Comparación de Modelos
Se comparan tres modelos entrenados en el mismo corpus:
- **Unigram**: Solo considera frecuencia de palabras individuales
- **Bigram**: Considera la palabra anterior
- **Trigram**: Considera las dos palabras anteriores

#### 3. Análisis de Sparsity
- Los modelos de mayor orden tienen más combinaciones posibles
- Muchas combinaciones nunca aparecen en el dataset (99.95%+ son ceros)
- Esto afecta la calidad de las predicciones

### 📊 Resultados Típicos

Basándose en el dataset Africa Galore (232 párrafos):

```
Dataset: 232 textos, ~31,000 tokens
Vocabulario: ~5,100 palabras únicas

Resultados esperados:
- Unigram Perplexity:   ~250-400
- Bigram Perplexity:    ~80-150
- Trigram Perplexity:   ~50-100
```

**Interpretación**:
- El trigram tiene menor perplexity → mejor predicción
- Pero también es más propenso a data sparsity
- Trade-off entre contexto y generalización

### 💡 Conceptos Clave Aprendidos

1. **Métrica de evaluación objetiva**: Perplexity permite comparar modelos cuantitativamente, no solo cualitativamente

2. **Trade-off contexto vs. datos**: Modelos con más contexto (mayor N) pueden predecir mejor, pero requieren mucho más datos para entrenar bien

3. **Data sparsity**:
   - Bigramas: 5,143 × 5,176 = 26M combinaciones posibles
   - Trigramas: 13,411 × 5,142 = 68M combinaciones posibles
   - Más del 99% nunca aparecen en el dataset

4. **Suavizado**: Cuando una secuencia no se ha visto, se asigna una probabilidad muy pequeña (1e-10) en lugar de 0 para evitar perplexity infinita

5. **Validación cruzada**: Es fundamental evaluar en datos NO vistos durante el entrenamiento

6. **Limitaciones de N-grams**:
   - Contexto muy limitado (solo N-1 palabras)
   - No capturan similitudes semánticas
   - Explosión combinatoria al aumentar N

### 🔧 Cómo Ejecutar

```bash
# Navegar al directorio
cd days_15_21_llm/day16_perplexity

# Compilar y ejecutar
cargo run --release

# Ver comparación detallada
cargo run --release -- --verbose
```

### 📈 Ejemplo de Salida

```
🚀 Evaluación de Modelos N-gram - Día 16
════════════════════════════════════════

📦 Dataset: 232 textos cargados
📊 Tokens totales: 31,234
📚 Vocabulario: 5,143 palabras

🔀 División datos:
   Train: 24,987 tokens (80%)
   Test:  6,247 tokens (20%)

════════════════════════════════════════
📊 RESULTADOS DE PERPLEXITY
════════════════════════════════════════

Modelo      | Perplexity | Contexto
------------|------------|----------
Unigram     |   342.15   | Ninguno
Bigram      |   127.83   | 1 palabra
Trigram     |    68.42   | 2 palabras

✅ Menor perplexity = Mejor modelo
🏆 Ganador: Trigram (68.42)

💡 El modelo trigram es 5x mejor que unigram
   en predecir la siguiente palabra.
```

### 🎓 ¿Por qué es Importante Perplexity?

1. **Evaluación estandarizada**: Permite comparar modelos diferentes de forma objetiva

2. **Métrica interpretable**: A diferencia de otras métricas complejas, perplexity tiene una interpretación intuitiva

3. **Conexión con información**: Matemáticamente relacionada con la entropía de Shannon (teoría de información)

4. **Predictor de calidad**: Correlaciona bien con la calidad percibida de generación de texto

5. **Base para modelos modernos**: Los transformers modernos también se evalúan con perplexity

### 🔬 Comparación con Transformers

Mientras que los modelos N-gram del Día 15-16 tienen:
- Perplexity: 50-400 (según N)
- Contexto: 1-2 palabras

Los modelos Transformer modernos (que veremos en Día 21) logran:
- Perplexity: 10-30 en el mismo dataset
- Contexto: 512-8192 tokens
- Captura de relaciones semánticas profundas

**Esto demuestra el poder de las arquitecturas neuronales modernas.**

### 📝 Notas de Desarrollo

Este es el **Día 16** de una serie de aprendizaje progresivo sobre modelos de lenguaje. El código está diseñado para ser educativo, con:

- Comentarios detallados en español
- Cálculos paso a paso de perplexity
- Comparaciones visuales claras
- Ejemplos con el dataset Africa Galore

**Próximos pasos**:
- Día 17: Tokenización BPE (mejor que space tokenizer)
- Día 18: Embeddings (representaciones vectoriales)
- Día 19-21: Redes neuronales → Transformer completo

### 🔗 Conexión con el Día 15

Este proyecto **depende directamente** del Día 15:
- Usa los mismos modelos N-gram
- Trabaja con el mismo dataset
- Agrega la capa de evaluación cuantitativa

**Diferencia clave**: Día 15 se enfoca en *construir* modelos, Día 16 en *evaluarlos*.

### 📚 Referencias Teóricas

- Perplexity es el exponencial de la entropía cruzada
- Formulación original en teoría de información (Shannon, 1948)
- Ampliamente usada en papers de NLP desde los años 90
- Sigue siendo métrica estándar para LLMs modernos

---

*Implementado en Rust para practicar tanto conceptos de ML como programación de sistemas.*
*Parte del plan maestro de aprendizaje de LLMs (días 15-21).*
