use std::f64::consts::E;

/// Estructura para representar una neurona simple
#[derive(Debug, Clone)]
 struct Neuron {
    pub synapses: f64,
    pub threshould: f64,
}

impl Neuron {
    pub fn new(value: f64) -> Self {
        Self {
            synapses: value,
            threshould: 0.0,
        }
    }
}

/// Función de activación Sigmoid
/// Fórmula: σ(x) = 1 / (1 + e^(-x))
/// Rango: (0, 1)
 fn sigmoid(z: f64) -> f64 {
    println!("Calculando sigmoid({:.4}):", z);
    println!("  e^(-z) = e^(-{:.4}) = {:.6}", z, E.powf(-z));
    println!("  1 + e^(-z) = {:.6}", 1.0 + E.powf(-z));

    1.0 / (1.0 + E.powf(-z))
}

/// Derivada de la función Sigmoid
/// Fórmula: σ'(x) = σ(x) * (1 - σ(x))
 fn sigmoid_derivative(x: f64) -> f64 {
    println!("Calculando sigmoid_derivative({:.4}):", x);
    
    let s = sigmoid(x);
    let one_minus_s = 1.0 - s;
    let derivative = s * one_minus_s;

    println!("  σ(x) = {:.6}", s);
    println!("  1 - σ(x) = {:.6}", one_minus_s);
    println!("  σ(x) * (1 - σ(x)) = {:.6}", derivative);

    derivative
}

/// Función de activación Tanh (Tangente Hiperbólica)
/// Fórmula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
/// Rango: (-1, 1)
 fn tanh_activation(x: f64) -> f64 {
    let exp_x = E.powf(x);
    let exp_neg_x = E.powf(-x);

    println!("Calculando tanh({:.4}):", x);
    println!("  e^x = {:.6}", exp_x);
    println!("  e^(-x) = {:.6}", exp_neg_x);
    println!("  Numerador: e^x - e^(-x) = {:.6}", exp_x - exp_neg_x);
    println!("  Denominador: e^x + e^(-x) = {:.6}", exp_x + exp_neg_x);

    let tanh = (exp_x - exp_neg_x) / (exp_x + exp_neg_x);
    println!("  tanh(x) = {:.6}", tanh);

    tanh
}

/// Derivada de la función Tanh
/// Fórmula: tanh'(x) = 1 - tanh²(x)
 fn tanh_derivative(x: f64) -> f64 {
    println!("Calculando tanh_derivative({:.4}):", x);

    let t = tanh_activation(x);
    let t_squared = t * t;
    let derivative = 1.0 - t_squared;

    println!("  tanh(x) = {:.6}", t);
    println!("  tanh²(x) = {:.6}", t_squared);
    println!("  1 - tanh²(x) = {:.6}", derivative);

    derivative

}

/// Función para aplicar activación a una neurona
 fn apply_activation(neuron: &mut Neuron, activation_fn: fn(f64) -> f64) {
    neuron.threshould = activation_fn(neuron.synapses);
}

/// Función para demostrar las diferencias entre ambas activaciones
 fn compare_activations() {
    println!("=== COMPARACIÓN DETALLADA: TANH vs SIGMOID ===\n");
    
    // Valores de prueba
    let test_values = vec![-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0];
    
    println!("| Input  | Sigmoid | Tanh   | Sig'   | Tanh'  |");
    println!("|--------|---------|--------|--------|--------|");
    
    for &x in &test_values {
        let sig_val = sigmoid(x);
        let tanh_val = tanh_activation(x);
        let sig_deriv = sigmoid_derivative(x);
        let tanh_deriv = tanh_derivative(x);
        
        println!("| {:6.1} | {:7.3} | {:6.3} | {:6.3} | {:6.3} |", 
                 x, sig_val, tanh_val, sig_deriv, tanh_deriv);
    }
    
    println!("\n=== ANÁLISIS DE CARACTERÍSTICAS ===");
    
    println!("\n📊 SIGMOID:");
    println!("• Rango de salida: (0, 1)");
    println!("• Siempre positiva");
    println!("• Centrada en 0.5");
    println!("• Problema del gradiente que desaparece para valores extremos");
    println!("• Útil para probabilidades y clasificación binaria");
    
    println!("\n📊 TANH:");
    println!("• Rango de salida: (-1, 1)");
    println!("• Centrada en 0 (media cercana a 0)");
    println!("• Simétrica respecto al origen");
    println!("• Gradientes más fuertes que Sigmoid");
    println!("• Mejor para capas ocultas en redes neuronales");
    
    println!("\n=== VENTAJAS Y DESVENTAJAS ===");
    
    println!("\n✅ VENTAJAS DE TANH:");
    println!("• Zero-centered: facilita el entrenamiento");
    println!("• Gradientes más fuertes en el rango medio");
    println!("• Convergencia más rápida");
    
    println!("\n❌ DESVENTAJAS DE TANH:");
    println!("• Sigue teniendo problema de gradiente desvaneciente");
    println!("• Computacionalmente más costosa que ReLU");
    
    println!("\n✅ VENTAJAS DE SIGMOID:");
    println!("• Interpretación probabilística clara");
    println!("• Suave y diferenciable");
    println!("• Ideal para la capa de salida en clasificación binaria");
    
    println!("\n❌ DESVENTAJAS DE SIGMOID:");
    println!("• No centrada en cero");
    println!("• Problema severo del gradiente desvaneciente");
    println!("• Saturación en los extremos");
}

/// Ejemplo práctico: Red neuronal simple con ambas activaciones
 fn neural_network_example() {
    println!("\n=== EJEMPLO: RED NEURONAL SIMPLE ===\n");
    
    // Crear neuronas de entrada
    let inputs = vec![-2.0, -0.5, 0.0, 1.5, 2.5];
    
    println!("Entrada: {:?}", inputs);
    
    // Aplicar Sigmoid
    println!("\n🔸 Con activación SIGMOID:");
    let sigmoid_neurons: Vec<Neuron> = inputs.iter()
        .map(|&x| {
            let mut neuron = Neuron::new(x);
            apply_activation(&mut neuron, sigmoid);
            neuron
        })
        .collect();
    
    for (i, neuron) in sigmoid_neurons.iter().enumerate() {
        println!("Neurona {}: {:.3} → {:.3}", 
                 i+1, neuron.synapses, neuron.threshould);
    }
    
    // Aplicar Tanh
    println!("\n🔸 Con activación TANH:");
    let tanh_neurons: Vec<Neuron> = inputs.iter()
        .map(|&x| {
            let mut neuron = Neuron::new(x);
            apply_activation(&mut neuron, tanh_activation);
            neuron
        })
        .collect();
    
    for (i, neuron) in tanh_neurons.iter().enumerate() {
        println!("Neurona {}: {:.3} → {:.3}", 
                 i+1, neuron.synapses, neuron.threshould);
    }
    
    // Calcular estadísticas
    let sigmoid_sum: f64 = sigmoid_neurons.iter()
        .map(|n| n.threshould)
        .sum();
    let sigmoid_mean = sigmoid_sum / sigmoid_neurons.len() as f64;
    
    let tanh_sum: f64 = tanh_neurons.iter()
        .map(|n| n.threshould)
        .sum();
    let tanh_mean = tanh_sum / tanh_neurons.len() as f64;
    
    println!("\n📈 ESTADÍSTICAS:");
    println!("Media Sigmoid: {:.3}", sigmoid_mean);
    println!("Media Tanh: {:.3}", tanh_mean);
    println!("\n💡 Observa cómo Tanh está más centrada en 0!");
}

/// Demostración de la importancia del centrado en cero
 fn zero_centered_demo() {
    println!("\n=== DEMOSTRACIÓN: IMPORTANCIA DEL CENTRADO EN CERO ===\n");
    
    let weights = vec![0.5, -0.3, 0.8, -0.2];
    let sigmoid_outputs = vec![0.8, 0.9, 0.7, 0.6]; // Típicas salidas sigmoid
    let tanh_outputs = vec![0.3, 0.5, -0.2, -0.1]; // Típicas salidas tanh
    
    println!("Pesos actuales: {:?}", weights);
    println!("Salidas Sigmoid: {:?}", sigmoid_outputs);
    println!("Salidas Tanh: {:?}", tanh_outputs);
    
    // Simular actualización de gradientes
    let learning_rate = 0.1;
    let error_gradient = 1.0; // Gradiente simplificado
    
    println!("\n🔄 Actualización de pesos (simplificada):");
    println!("Con Sigmoid (siempre positivas):");
    for (i, &w) in weights.iter().enumerate() {
        let gradient = error_gradient * sigmoid_outputs[i];
        let new_weight = w - learning_rate * gradient;
        println!("  Peso[{}]: {:.3} → {:.3} (Δ = {:.3})", 
                 i, w, new_weight, -learning_rate * gradient);
    }
    
    println!("\nCon Tanh (puede ser negativas):");
    for (i, &w) in weights.iter().enumerate() {
        let gradient = error_gradient * tanh_outputs[i];
        let new_weight = w - learning_rate * gradient;
        println!("  Peso[{}]: {:.3} → {:.3} (Δ = {:.3})", 
                 i, w, new_weight, -learning_rate * gradient);
    }
    
    println!("\n💡 Con Sigmoid, todos los gradientes tienen el mismo signo");
    println!("   → Actualizaciones sesgadas");
    println!("💡 Con Tanh, los gradientes pueden tener signos diferentes");
    println!("   → Actualizaciones más balanceadas");
}

fn main() {
    println!("🧠 RETO DE REDES NEURONALES: TANH vs SIGMOID en Rust");
    println!("================================================\n");
    
    // Ejecutar todas las demostraciones
    compare_activations();
    neural_network_example();
    zero_centered_demo();
    
    println!("\n=== RESUMEN CLAVE ===");
    println!("🎯 Usa TANH para capas ocultas (mejor gradiente, centrada en 0)");
    println!("🎯 Usa SIGMOID para salida de clasificación binaria");
    println!("🎯 En práctica moderna, considera ReLU para capas ocultas");
    
    println!("\n✅ ¡Reto completado! Has aprendido las diferencias clave entre Tanh y Sigmoid.");
}