# 🚀 Small Transformer LLM

Día 21 del Desafío de 21 Días - Transformer Completo

## Descripción del Proyecto

Implementación completa de un Small Language Model basado en arquitectura Transformer.

## Características Implementadas

- [ ] Transformer encoder completo
- [ ] Transformer decoder completo
- [ ] Layer Normalization
- [ ] Feed-forward network
- [ ] Sistema de entrenamiento completo
- [ ] Generación de texto autoregresiva
- [ ] Checkpoint saving/loading
- [ ] Métricas de evaluación (loss, perplexity)

## Cómo Ejecutar

```bash
cd days_15_21_llm/day21_small_transformer

# Entrenar modelo
cargo run --release -- train

# Generar texto
cargo run --release -- generate "Abeni was"
```

## Arquitectura

- Vocab size: 5000-8000 tokens (BPE)
- Embedding dim: 128
- Hidden dim: 256
- Num heads: 4
- Num layers: 2 (encoder) + 2 (decoder)
- Total params: ~3-4M

## Dataset

Africa Galore (232 párrafos)

---

**Nota**: Este proyecto es el objetivo final del plan maestro de aprendizaje de LLMs (días 15-21).
