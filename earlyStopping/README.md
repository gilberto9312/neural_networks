# 🧠 Día 11: Early Stopping (Parada Temprana)

Una implementación en Rust que demuestra cómo utilizar **Early Stopping** para prevenir el sobreajuste y optimizar el tiempo de entrenamiento en una red neuronal.

## 🎯 Descripción

Este proyecto implementa una red neuronal desde cero para resolver el problema XOR. El objetivo principal es mostrar cómo la técnica de **Early Stopping** puede detener el entrenamiento de manera inteligente cuando el rendimiento en un conjunto de validación deja de mejorar, evitando así el sobreajuste y ahorrando ciclos de computación.

Se comparan varios escenarios para evaluar el impacto de esta técnica:
1.  Entrenamiento sin Early Stopping (línea de base).
2.  Entrenamiento con Early Stopping "conservador" (paciencia alta).
3.  Entrenamiento con Early Stopping "agresivo" (paciencia baja).
4.  Combinación de Early Stopping con otras técnicas de regularización como **L2** y **Dropout**.

## 🏗️ Arquitectura de la Red

-   **Capa de entrada**: 2 neuronas
-   **Capa oculta**: 8 neuronas (función de activación: Sigmoid)
-   **Capa de salida**: 1 neurona (función de activación: Sigmoid)
-   **Inicialización**: Pesos aleatorios con una semilla fija para garantizar la reproducibilidad de los experimentos.
-   **Optimizador**: Se utiliza **Adam** en todos los escenarios por su eficiencia y robustez.

## 🚀 Técnica: Early Stopping

El Early Stopping es una forma de regularización que monitoriza el rendimiento del modelo en un conjunto de datos de validación.

### ¿Cómo funciona?

1.  **Monitorización**: Después de cada época, se calcula el error (pérdida) del modelo en el conjunto de validación.
2.  **Paciencia (`patience`)**: Se define un número de épocas a esperar. Si el error de validación no mejora (es decir, no disminuye en una cantidad `min_delta`) durante este número de épocas, el entrenamiento se detiene.
3.  **Restauración**: Una vez detenido, el modelo no se queda con los últimos pesos, sino que se restaura al estado donde obtuvo el **mejor error de validación**.

```rust
// Estructura principal de EarlyStopping
struct EarlyStopping {
    patience: usize,           // Épocas a esperar sin mejora
    min_delta: f64,            // Mínima mejora considerada significativa
    wait_counter: usize,       // Contador de épocas sin mejora
    best_loss: f64,            // Mejor pérdida de validación encontrada
    best_weights: Option<NetworkWeights>, // Mejores pesos guardados
}
```

### Ventajas
-   **Previene el Sobreajuste**: Detiene el entrenamiento antes de que el modelo empiece a memorizar el ruido de los datos de entrenamiento.
-   **Eficiencia**: Ahorra tiempo al no ejecutar épocas que no aportan valor.
-   **Mejor Generalización**: Tiende a producir modelos que funcionan mejor con datos no vistos.

## 🛠️ Características Técnicas

-   **Implementación Modular**: La lógica de Early Stopping está encapsulada en su propia `struct`, haciéndola reutilizable.
-   **Regularización Combinada**: El código está preparado para usar Early Stopping junto con regularizadores L1, L2 y Dropout.
-   **Visualización**: Se utiliza la librería `plotters` para generar una gráfica de la curva de entrenamiento (`training_curve.png`), mostrando la evolución del error de entrenamiento vs. el de validación.
-   **Sin dependencias externas (casi)**: La lógica de la red neuronal y los optimizadores está implementada desde cero. `plotters` es la única dependencia principal.

## 📊 Salida del Programa

```
🧠 RED NEURONAL CON EARLY STOPPING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Datos de entrenamiento (función XOR):
   [0.0, 0.0] → [0.0]
   [0.0, 1.0] → [1.0]
   [1.0, 0.0] → [1.0]
   [1.1, 1.0] → [0.0]
🎯 Épocas máximas: 10000

🔹 Entrenando con Early Stopping: Sin Early Stopping (2000 épocas)
  📊 Época 0: Val=0.575620, Mejor=inf (época 0), Espera=1/100
  🎯 MEJORA Época 1: Val=0.574210, Mejor=0.574210 (época 1), Espera=0/100
  ...
  → 📈 ÉPOCAS COMPLETAS en 2000 épocas: Train=0.000000, Val=0.000000 (pesos restaurados)

🔹 Entrenando con Early Stopping: Early Stopping Conservador
  ...
  🎯 MEJORA Época 1393: Val=0.000000, Mejor=0.000000 (época 1393), Espera=0/200
🛑 Early Stopping activado en época 1593 (sin mejora por 200 épocas)
   Mejor validación: 0.000000 en época 1393
🔄 Pesos restaurados al mejor modelo (época 1393)
  → 🛑 PARADA TEMPRANA en 1594 épocas: Train=0.000000, Val=0.000000 (pesos restaurados)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 RESUMEN DE EARLY STOPPING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔹 Sin Early Stopping 🛑 → Épocas: 624, Train: 0.000296, Val: 0.000296
🔹 Early Stop Conservador 🛑 → Épocas: 496, Train: 0.000982, Val: 0.000982
🔹 Early Stop Agresivo 🛑 → Épocas: 507, Train: 0.000397, Val: 0.000397
🔹 Early Stop + L2 🛑 → Épocas: 467, Train: 0.006809, Val: 0.006809
🔹 Early Stop + Dropout 🛑 → Épocas: 798, Train: 0.001581, Val: 0.001581

🏆 MÁS EFICIENTE: Sin Early Stopping con error 0.000296 en 624 épocas
   ✨ Se benefició del Early Stopping

✅ Demostración de Early Stopping completada!
💡 Early Stopping ayuda a:
   • Evitar sobreajuste (overfitting)
   • Reducir tiempo de entrenamiento
   • Encontrar el modelo con mejor generalización

## 📈 Interpretación de Resultados

-   **Curva de Entrenamiento**: Abre el archivo `training_curve.png` para visualizar el comportamiento. Notarás que el error de validación (azul) puede estancarse o incluso subir mientras el de entrenamiento (rojo) sigue bajando. Early Stopping actúa en ese momento.
-   **Eficiencia**: Los escenarios con Early Stopping generalmente terminan en menos épocas que el máximo configurado, demostrando su capacidad para ahorrar tiempo.
-   **Rendimiento**: El error de validación final suele ser igual o mejor con Early Stopping, ya que se seleccionan los pesos del punto óptimo de generalización.

![Curva de Entrenamiento](training_curve.png)

## 🔧 Personalización

### Modificar Parámetros de Early Stopping
En la función `main()`:
```rust
// Escenario con Early Stopping Agresivo
let early_stop = EarlyStopping::new(
    50,      // patience: esperar 50 épocas
    0.0001   // min_delta: mejora mínima requerida
);
```

### Modificar Arquitectura de Red
En `NeuralNetwork::new_with_seed()`:
```rust
let network = NeuralNetwork::new_with_seed(
    2,  // Neuronas de entrada
    8,  // Neuronas ocultas
    1,  // Neuronas de salida
    0.1, // Learning rate
    42   // Semilla
);
```

## 📚 Conceptos Aprendidos

-   **Regularización**: Early Stopping como técnica para combatir el sobreajuste.
-   **Sobreajuste (Overfitting)**: Fenómeno donde un modelo aprende tan bien los datos de entrenamiento que pierde la capacidad de generalizar a nuevos datos.
-   **Conjunto de Validación**: La importancia de tener un conjunto de datos separado para evaluar el rendimiento del modelo de forma objetiva durante el entrenamiento.
-   **Gestión de Estado**: Cómo guardar y restaurar el "mejor" estado de un modelo durante un proceso iterativo.

---

**¿Encontraste este proyecto útil?** ⭐ ¡Dale una estrella al repositorio!