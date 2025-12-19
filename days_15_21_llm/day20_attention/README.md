# ⚡ Mecanismo de Atención

Día 20 del Desafío de 21 Días - Mecanismo de Atención

## Descripción del Proyecto

Implementación del mecanismo de atención, componente fundamental de los Transformers.

## Características Implementadas

- [ ] Atención básica (Q, K, V)
- [ ] Scaled Dot-Product Attention
- [ ] Multi-Head Attention
- [ ] Positional Encoding (sinusoidal)
- [ ] Visualización de pesos de atención (heatmaps)

## Cómo Ejecutar

```bash
cd days_15_21_llm/day20_attention
cargo run --release
```

## Conceptos Teóricos

### Scaled Dot-Product Attention
```
Attention(Q,K,V) = softmax(QK^T / √d_k)V
```

### Multi-Head Attention
Permite al modelo atender a información de diferentes subespacios de representación.

---

**Nota**: Este proyecto es parte del plan maestro de aprendizaje de LLMs (días 15-21).
