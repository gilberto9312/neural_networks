
fn main() {
    println!("=== Red Neuronal en Rust ===\n");

    // 1. Crear neuronas con diferentes configuraciones
    println!("1. Creando neuronas:");
    
    // Neurona con valores por defecto
    let neuron1 = neuron(None, None);
    println!("Neurona 1 (por defecto): {:?}", neuron1);
    
    // Neurona con threshold personalizado
    let neuron2 = neuron(None, Some(0.5));
    println!("Neurona 2 (threshold 0.5): {:?}", neuron2);
    
    // Neurona con sinapsis iniciales
    let neuron3 = neuron(Some(vec![0.1, 0.2, 0.3]), Some(0.8));
    println!("Neurona 3 (con datos): {:?}\n", neuron3);

    // 2. Crear sinapsis individuales
    println!("2. Creando sinapsis:");
    
    let syn1 = synapse(Some(0.6), Some(1.5));  // weight=0.6, value=1.5
    let syn2 = synapse(Some(0.4), Some(2.0));  // weight=0.4, value=2.0
    let syn3 = synapse(None, Some(1.0));        // weight=0.8, value=0.0 (default)
    
    println!("Sinapsis 1: {:?}", syn1);
    println!("Sinapsis 2: {:?}", syn2);
    println!("Sinapsis 3: {:?}", syn3);

    // 3. Probar la función shouldTrigger
    println!("3. Pruebas de activación neuronal:");
    
    let synapses_test = vec![syn1, syn2, syn3];
    
    // Calcular la suma manualmente para verificar
    let manual_sum = (0.6 * 1.5) + (0.4 * 2.0) + (0.1 * 1.0);
    println!("Suma manual: {}", manual_sum); // 0.9 + 0.8 + 0.1 = 1.8
    
    // Probar diferentes thresholds
    println!("¿Se activa con threshold 1.0? {}", should_trigger(1.0, &synapses_test)); // true
    println!("¿Se activa con threshold 1.5? {}", should_trigger(1.5, &synapses_test)); // true
    println!("¿Se activa con threshold 2.0? {}", should_trigger(2.0, &synapses_test)); // false
    
    // 4. Simulación de una pequeña red neuronal
    println!("\n4. Simulación de red neuronal:");
    
    // Crear múltiples neuronas
    let neurons = vec![
        neuron(None, Some(0.5)),           // Neurona sensible (threshold bajo)
        neuron(None, Some(1.5)),           // Neurona moderada
        neuron(None, Some(2.5)),           // Neurona insensible (threshold alto)
    ];
    
    // Crear diferentes conjuntos de entrada
    let inputs = vec![
        vec![
            synapse(Some(0.3), Some(1.0)),
            synapse(Some(0.4), Some(0.8)),
        ],
        vec![
            synapse(Some(0.7), Some(2.0)),
            synapse(Some(0.5), Some(1.5)),
        ],
        vec![
            synapse(Some(0.9), Some(3.0)),
            synapse(Some(0.8), Some(2.5)),
        ],
    ];
    
    // Evaluar cada neurona con cada conjunto de entrada
    for (i, input_set) in inputs.iter().enumerate() {
        println!("\nConjunto de entrada {}:", i + 1);
        let input_sum: f32 = input_set.iter()
            .map(|s| s.weight * s.value)
            .sum();
        println!("  Suma de entrada: {:.2}", input_sum);
        
        for (j, neuron) in neurons.iter().enumerate() {
            let activated = should_trigger(neuron.threshould, input_set);
            println!("  Neurona {} (threshold {:.1}): {}", 
                    j + 1, neuron.threshould, if activated { "ACTIVADA" } else { "inactiva" });
        }
    }
    
    // 5. Ejemplo práctico: Detector de patrones
    println!("\n5. Ejemplo: Detector de patrones alto/bajo:");
    
    let pattern_detector = neuron(None, Some(1.2)); // Threshold medio
    
    let patterns = vec![
        ("Patrón bajo", vec![
            synapse(Some(0.2), Some(1.0)),
            synapse(Some(0.3), Some(0.5)),
        ]),
        ("Patrón medio", vec![
            synapse(Some(0.5), Some(1.5)),
            synapse(Some(0.4), Some(1.0)),
        ]),
        ("Patrón alto", vec![
            synapse(Some(0.8), Some(2.0)),
            synapse(Some(0.7), Some(1.8)),
        ]),
    ];
    
    for (pattern_name, pattern_synapses) in patterns {
        let detected = should_trigger(pattern_detector.threshould, &pattern_synapses);
        let sum: f32 = pattern_synapses.iter()
            .map(|s| s.weight * s.value)
            .sum();
        println!("{}: suma={:.2}, detectado={}", 
                pattern_name, sum, if detected { "SÍ" } else { "NO" });
    }
}

#[derive(Debug, Clone)]
struct Neuron {
    synapses: Vec<f32>,
    threshould: f32
} 

#[derive(Debug, Clone)]
struct Synapse {
    weight: f32,
    value: f32
}



impl Neuron {
    fn new (synapses: Option<Vec<f32>>, threshould:Option<f32>) -> Self{
        Self { 
            synapses: synapses.unwrap_or_else(Vec::new),
            threshould: threshould.unwrap_or(1.0)
        }
    }
}


impl Synapse {
    fn new (weight:Option<f32>, value: Option<f32>) -> Self {
        Self{
            weight: weight.unwrap_or(0.1),
            value: value.unwrap_or(0.0)
        }
    }
}


fn neuron(synapses: Option<Vec<f32>>, threshould:Option<f32>) -> Neuron {
    Neuron::new(synapses, threshould)
}

fn synapse(weight:Option<f32>, value: Option<f32>) -> Synapse {
    Synapse::new(weight, value)
}

fn should_trigger (threshould:f32, synapses: &[Synapse]) -> bool {
    let sum: f32 = synapses
        .iter()
        .map(|synapse|synapse.weight * synapse.value)
        .sum();
    sum >= threshould
}
