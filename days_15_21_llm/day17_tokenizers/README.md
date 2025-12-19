# 🔤 Tokenizadores desde Cero

Día 17 del Desafío de 21 Días - Tokenización y Preprocesamiento

## Descripción del Proyecto

Implementación de tokenizadores (caracteres, palabras, BPE) para preprocesar texto antes de entrenar modelos de lenguaje.

## Características Implementadas

- [ ] Tokenizador por caracteres
- [ ] Tokenizador por palabras (whitespace)
- [ ] Tokenizador BPE (Byte Pair Encoding)
- [ ] Vocabulario y mapeo token↔ID
- [ ] Padding y truncamiento
- [ ] Tokens especiales (PAD, UNK, BOS, EOS)

## Cómo Ejecutar

```bash
cd days_15_21_llm/day17_tokenizers
cargo run --release
```

## Conceptos Teóricos

### Byte Pair Encoding (BPE)
Algoritmo de compresión que encuentra los pares de bytes más frecuentes y los reemplaza iterativamente.

---

**Nota**: Este proyecto es parte del plan maestro de aprendizaje de LLMs (días 15-21).
