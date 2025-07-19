use std::f64::consts::E;
fn main() {
    println!("🧠 REDES NEURONALES - DÍA: FUNCIÓN SIGMOID");
    println!("===========================================");
    println!("Objetivo: Entender cómo sigmoid transforma salidas lineales en no-lineales\n");
    
    // 1. Analizar la función sigmoid teóricamente
    analizar_sigmoid();
    
    // 2. Comparar salidas lineales vs no-lineales en práctica
    comparar_lineal_vs_nolineal();
    
    println!("\n🎯 PUNTOS CLAVE APRENDIDOS:");
    println!("1. Salida lineal: z = Σ(wi * xi) + b - puede ser cualquier valor real");
    println!("2. Sigmoid comprime valores reales al rango (0,1)");
    println!("3. Sigmoid introduce no-linealidad esencial para redes neuronales");
    println!("4. Valores grandes (+/-) se saturan cerca de 1/0");
    println!("5. La curva suave permite gradientes para entrenamiento");
    
    println!("\n✅ ¡Sigmoid domado! Listo para el siguiente reto.");
}


#[derive(Debug, Clone)]
struct Neuron {
    synapses: Vec<f64>,
    threshold: f64
} 

impl Neuron {
    fn new(synapses : Vec<f64>, threshold: f64) -> Self {
        Neuron {synapses, threshold}
    }

    fn salida_lineal(&self, entradas: &[f64]) -> f64 {
        println!("\n === calculando salida lineal");
        //producto punto: suma de (entrada_i *
        let suma_ponderada: f64 = self.synapses.iter()
            .zip(entradas.iter())
            .enumerate()
            .map(|(i, (&synapse, &entrada)) | {
            let producto = synapse * entrada;
            println!("entrada[{}] = {:.2} * peso[{}] = {:.2} = {:.2}", 
                        i, entrada, i, synapse, producto);
            producto
            })
            .sum();
        let salida_lineal = suma_ponderada + self.threshold;
        println!("Suma ponderada: {:.4}", suma_ponderada);
        println!("threshold: {:.4}", self.threshold);
        println!("Salida lineal (z): {:.4}", salida_lineal);
        
        salida_lineal
    }
    fn salida_con_sigmoid(&self, entradas: &[f64]) -> f64 {
        let z = self.salida_lineal(entradas);
        let sigmoid_output = sigmoid(z);
        
        println!("\n=== APLICANDO FUNCIÓN SIGMOID ===");
        println!("z (entrada lineal): {:.4}", z);
        println!("sigmoid(z): {:.6}", sigmoid_output);
        
        sigmoid_output
    }
}

fn sigmoid(z: f64) ->f64{
    println!("Calculando sigmoid({:.4}):", z);
    println!("  e^(-z) = e^(-{:.4}) = {:.6}", z, E.powf(-z));
    println!("  1 + e^(-z) = {:.6}", 1.0 + E.powf(-z));
    
    1.0 / (1.0 + E.powf(-z))
}

fn analizar_sigmoid(){
    println!("\n🔍 ANÁLISIS DE LA FUNCIÓN SIGMOID");
    println!("=====================================");
    println!("La función sigmoid σ(z) = 1/(1 + e^(-z)) tiene estas propiedades:");
    println!("• Rango: (0, 1) - nunca llega exactamente a 0 o 1");
    println!("• Forma de S suave");
    println!("• Derivable en todos los puntos");
    println!("• σ(0) = 0.5 (punto medio)");
    println!("• Cuando z → +∞, σ(z) → 1");
    println!("• Cuando z → -∞, σ(z) → 0");
    
    println!("\nEjemplos de valores:");
    let valores_pruebas = vec![-10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0];
    for z in valores_pruebas {
        println!("sigmoid({:4.1}) = {:.6}", z, sigmoid(z));
    }
}

fn comparar_lineal_vs_nolineal(){
    println!("\n📊 COMPARACIÓN: LINEAL vs NO-LINEAL");
    println!("=====================================");
    
    // Crear una neurona simple
    let neurona = Neuron::new(vec![0.5, -0.3, 0.8], 0.1);
    
    println!("Neurona configurada:");
    println!("  Synapse: {:?}", neurona.synapses);
    println!("  Threshold: {}", neurona.threshold);
    
    // Probar con diferentes entradas
    let conjuntos_entrada = vec![
        vec![1.0, 2.0, -1.0],
        vec![-2.0, 1.5, 3.0],
        vec![0.0, 0.0, 0.0],
        vec![10.0, -5.0, 2.0],
    ];
    
    println!("\n📈 RESULTADOS:");
    println!("{:<20} | {:^15} | {:^15}", "Entradas", "Salida Lineal", "Salida Sigmoid");
    println!("{}", "-".repeat(55));
    
    for (i, entradas) in conjuntos_entrada.iter().enumerate() {
        println!("\n --- Conjunto {} ---",i +1);
        let lineal = neurona.salida_lineal(entradas);
        let no_lineal = neurona.salida_con_sigmoid(entradas);
        println!("{:<20} | {:^15.4} | {:^15.6}", 
                format!("{:?}", entradas), lineal, no_lineal);
        
        if lineal > 0.0 {
            println!("💡 Entrada positiva → Sigmoid tiende hacia 1.0");
        } else if lineal < 0.0 {
            println!("💡 Entrada negativa → Sigmoid tiende hacia 0.0");
        } else {
            println!("💡 Entrada cero → Sigmoid = 0.5 exacto");
        }

    }
}