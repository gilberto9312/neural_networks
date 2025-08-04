# 🧠 Día 10 - Visualización de Resultados en Rust

En el día 10 de nuestra serie de ejercicios para aprender inteligencia artificial con Rust, hemos dado un paso muy importante: **visualizar los resultados del entrenamiento de nuestras redes neuronales**.

## 🎯 ¿Qué hicimos?

Hoy integramos el crate [`plotters`](https://crates.io/crates/plotters) para graficar dos aspectos fundamentales del proceso de entrenamiento:

- **Error de entrenamiento vs error de validación**
- **Convergencia de los modelos (si aprendieron o no)**

Estas gráficas fueron generadas luego de ejecutar múltiples experimentos con distintos tipos de regularización: sin regularización, L1, L2, Dropout, y combinaciones de estas.

## 📊 ¿Por qué es importante graficar?

La visualización de resultados es clave en todo proyecto de machine learning, por varias razones:

1. **Identificación de sobreajuste o subajuste:**  
   Ver cómo evoluciona el error de validación frente al de entrenamiento permite identificar cuándo un modelo está aprendiendo demasiado bien los datos de entrenamiento (overfitting) o no está aprendiendo lo suficiente (underfitting).

2. **Comparación objetiva entre modelos:**  
   No basta con decir que un modelo “parece aprender mejor”. Ver curvas y tasas de convergencia nos da evidencia clara y cuantificable de cuál estrategia funciona mejor.

3. **Validación de convergencia:**  
   Algunas combinaciones de hiperparámetros o técnicas de regularización pueden impedir que el modelo converja. La gráfica de convergencia (de tipo barra) nos muestra cuántos modelos lograron aprender dentro del número de épocas dado.

4. **Iteración basada en evidencia:**  
   Con los resultados visualizados, podemos tomar decisiones informadas para ajustar parámetros, cambiar arquitecturas, o aplicar otras técnicas.

## 🖼️ Ejemplo de la gráfica generada

Al final del entrenamiento, generamos una imagen (`training_metrics.png`) que contiene:

- 📉 Una gráfica de línea con la evolución del error por época.
- 📌 Una gráfica de barras con la cantidad de modelos que convergieron frente a los que no.

Estas visualizaciones no solo nos ayudan a entender el comportamiento de la red, sino que también documentan el progreso de nuestro aprendizaje.

---

> 📍 Este es un paso fundamental antes de continuar con arquitecturas más complejas o datasets más grandes. ¡Visualizar antes de optimizar!

