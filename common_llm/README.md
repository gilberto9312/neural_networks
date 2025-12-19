# 📚 Common LLM Library

Librería compartida para los proyectos de LLM (días 15-21).

## Descripción

Contiene código común reutilizable entre todos los proyectos de LLM:
- Funciones de activación
- Funciones de pérdida (loss)
- Optimizadores (Adam, SGD)
- Cargador de datasets
- Métricas (perplexity, accuracy)

## Uso

Agregar como dependencia en los proyectos:

```toml
[dependencies]
common_llm = { path = "../../common_llm" }
```

---

**Nota**: Esta librería es parte del plan maestro de aprendizaje de LLMs (días 15-21).
