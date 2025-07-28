use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

// ================ PSEUDOCÓDIGO GENERAL ================
/*
ARQUITECTURA GENERAL:

1. Definir 3 structs de optimizadores:
   - MomentumOptimizer: velocidad = β * velocidad + gradiente
   - RMSpropOptimizer: cache = α * cache + (1-α) * gradiente²; actualización = gradiente / sqrt(cache + ε)
   - AdamOptimizer: momento1 = β1 * momento1 + (1-β1) * gradiente
                    momento2 = β2 * momento2 + (1-β2) * gradiente²
                    corrección de sesgo + actualización

2. Cada optimizador tiene su propia red neuronal idéntica (mismos pesos iniciales)

3. Entrenar en 3 hilos paralelos:
   Hilo 1: Red + Momentum
   Hilo 2: Red + RMSprop  
   Hilo 3: Red + Adam

4. Usar channels para comunicar resultados:
   - Cuándo termina cada optimizador
   - Error final de cada uno
   - Evolución del error durante entrenamiento

5. Al final: mostrar cuál fue el ganador
*/

// ================ ESTRUCTURA BASE DE LA RED ================
#[derive(Debug, Clone)]
struct NeuralNetwork {
    weights_input_hidden: Vec<Vec<f64>>,
    bias_hidden: Vec<f64>,
    weights_hidden_output: Vec<Vec<f64>>,
    bias_output: Vec<f64>,
    learning_rate: f64,
}

impl NeuralNetwork {
    // Constructor con seed para reproducibilidad
    fn new_with_seed(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64, seed: u64) -> Self {
        let mut rng_state = seed;
        
        // Función simple de random (Linear Congruential Generator)
        //generador de números aleatorios manual llamado Linear Congruential Generator (LCG), 
        //que es una forma muy antigua (pero sencilla) de generar números pseudoaleatorios.
        //Xₙ₊₁ = (a * Xₙ + c) mod m
        let mut next_random = || {
            //si hay desbordamiento, no falles, solo "envuélvelo"
            //(wrap around, como un reloj de 12 horas).
            rng_state = rng_state
                .wrapping_mul(1103515245)
                //Suma una constante pequeña para "desordenar" un poco más el resultado (esto evita ciclos cortos en la secuencia pseudoaleatoria).
                .wrapping_add(12345);
                //Esto es una forma de normalizar el número aleatorio:
                // rng_state / 65536 elimina los 16 bits menos significativos → reduce la granularidad (como hacer zoom out).
                // % 32768 asegura que el número aleatorio esté entre 0 y 32767.
                // Esto emula cómo se hacía la generación de aleatorios en C con rand().
            (rng_state / 65536) % 32768
        };
        
        let mut random_weight = || ((next_random() as f64) / 32768.0) * 2.0 - 1.0; // Entre -1 y 1
        
        // Inicializar pesos entrada → oculta
        let mut weights_input_hidden = Vec::new();
        for _i in 0..input_size {
            let mut row = Vec::new();
            for _j in 0..hidden_size {
                row.push(random_weight() * 0.5); // Escalar para estabilidad
            }
            weights_input_hidden.push(row);
        }
        
        // Inicializar bias capa oculta
        let mut bias_hidden = Vec::new();
        for _i in 0..hidden_size {
            bias_hidden.push(random_weight() * 0.1);
        }
        
        // Inicializar pesos oculta → salida
        let mut weights_hidden_output = Vec::new();
        for _i in 0..hidden_size {
            let mut row = Vec::new();
            for _j in 0..output_size {
                row.push(random_weight() * 0.5);
            }
            weights_hidden_output.push(row);
        }
        
        // Inicializar bias capa salida
        let mut bias_output = Vec::new();
        for _i in 0..output_size {
            bias_output.push(random_weight() * 0.1);
        }
        
        Self {
            weights_input_hidden,
            bias_hidden,
            weights_hidden_output,
            bias_output,
            learning_rate,
        }
    }
    
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    
    fn forward(&self, inputs: &[f64]) -> (Vec<f64>, Vec<f64>) {
        // Capa oculta
        let mut hidden = Vec::new();
        for h in 0..self.bias_hidden.len() {
            let mut sum = self.bias_hidden[h];
            for (i, &input) in inputs.iter().enumerate() {
                sum += input * self.weights_input_hidden[i][h];
            }
            hidden.push(Self::sigmoid(sum));
        }
        
        // Capa de salida
        let mut output = Vec::new();
        for o in 0..self.bias_output.len() {
            let mut sum = self.bias_output[o];
            for (h, &hidden_val) in hidden.iter().enumerate() {
                sum += hidden_val * self.weights_hidden_output[h][o];
            }
            output.push(Self::sigmoid(sum));
        }
        
        (hidden, output)
    }
    
    // Calcular gradientes (sin actualizar pesos)
    fn compute_gradients(&self, inputs: &[f64], targets: &[f64]) -> (NetworkGradients, f64) {
        let (hidden, output) = self.forward(inputs);
        
        // Errores de salida
        let mut output_errors = Vec::new();
        let mut total_error = 0.0;
        
        for (&out, &target) in output.iter().zip(targets.iter()) {
            let error = target - out;
            total_error += 0.5 * error.powi(2);
            let delta = -error * out * (1.0 - out);
            output_errors.push(delta);
        }
        
        // Errores de capa oculta
        let mut hidden_errors = Vec::new();
        for h in 0..hidden.len() {
            let mut error_sum = 0.0;
            for (o, &output_error) in output_errors.iter().enumerate() {
                error_sum += output_error * self.weights_hidden_output[h][o];
            }
            let delta = error_sum * hidden[h] * (1.0 - hidden[h]);
            hidden_errors.push(delta);
        }
        
        // Construir gradientes
        let mut grad_weights_hidden_output = vec![vec![0.0; self.weights_hidden_output[0].len()]; self.weights_hidden_output.len()];
        for h in 0..hidden.len() {
            for (o, &output_error) in output_errors.iter().enumerate() {
                grad_weights_hidden_output[h][o] = output_error * hidden[h];
            }
        }
        
        let grad_bias_output = output_errors.clone();
        
        let mut grad_weights_input_hidden = vec![vec![0.0; self.weights_input_hidden[0].len()]; self.weights_input_hidden.len()];
        for (i, &input) in inputs.iter().enumerate() {
            for (h, &hidden_error) in hidden_errors.iter().enumerate() {
                grad_weights_input_hidden[i][h] = hidden_error * input;
            }
        }
        
        let grad_bias_hidden = hidden_errors.clone();
        
        let gradients = NetworkGradients {
            grad_weights_input_hidden,
            grad_bias_hidden,
            grad_weights_hidden_output,
            grad_bias_output,
        };
        
        (gradients, total_error)
    }
    
 
}

// ================ ESTRUCTURA PARA GRADIENTES ================
#[derive(Debug, Clone)]
struct NetworkGradients {
    grad_weights_input_hidden: Vec<Vec<f64>>,
    grad_bias_hidden: Vec<f64>,
    grad_weights_hidden_output: Vec<Vec<f64>>,
    grad_bias_output: Vec<f64>,
}

// ================ OPTIMIZADOR MOMENTUM ================
/*
MOMENTUM EXPLICACIÓN:
- Mantiene una "velocidad" (momentum) que acumula gradientes pasados
- Fórmula: velocity = β * velocity_anterior + gradiente_actual
- Actualización: peso = peso - learning_rate * velocity
- β típicamente 0.9 (90% de la velocidad anterior, 10% del gradiente actual)
- Ventaja: suaviza oscilaciones, ayuda a escapar mínimos locales
*/
struct MomentumOptimizer {
    velocity_weights_input_hidden: Vec<Vec<f64>>,
    velocity_bias_hidden: Vec<f64>,
    velocity_weights_hidden_output: Vec<Vec<f64>>,
    velocity_bias_output: Vec<f64>,
    beta: f64, // Factor de momentum
}

impl MomentumOptimizer {
    fn new(network: &NeuralNetwork, beta: f64) -> Self {
        let velocity_weights_input_hidden = vec![vec![0.0; network.weights_input_hidden[0].len()]; network.weights_input_hidden.len()];
        let velocity_bias_hidden = vec![0.0; network.bias_hidden.len()];
        let velocity_weights_hidden_output = vec![vec![0.0; network.weights_hidden_output[0].len()]; network.weights_hidden_output.len()];
        let velocity_bias_output = vec![0.0; network.bias_output.len()];
        
        Self {
            velocity_weights_input_hidden,
            velocity_bias_hidden,
            velocity_weights_hidden_output,
            velocity_bias_output,
            beta,
        }
    }
    
    fn update(&mut self, network: &mut NeuralNetwork, gradients: &NetworkGradients) {
        // Actualizar velocidades y pesos entrada → oculta
        for i in 0..network.weights_input_hidden.len() {
            for j in 0..network.weights_input_hidden[i].len() {
                self.velocity_weights_input_hidden[i][j] = 
                    self.beta * self.velocity_weights_input_hidden[i][j] + gradients.grad_weights_input_hidden[i][j];
                network.weights_input_hidden[i][j] -= network.learning_rate * self.velocity_weights_input_hidden[i][j];
            }
        }
        
        // Bias oculta
        for i in 0..network.bias_hidden.len() {
            self.velocity_bias_hidden[i] = self.beta * self.velocity_bias_hidden[i] + gradients.grad_bias_hidden[i];
            network.bias_hidden[i] -= network.learning_rate * self.velocity_bias_hidden[i];
        }
        
        // Pesos oculta → salida
        for i in 0..network.weights_hidden_output.len() {
            for j in 0..network.weights_hidden_output[i].len() {
                self.velocity_weights_hidden_output[i][j] = 
                    self.beta * self.velocity_weights_hidden_output[i][j] + gradients.grad_weights_hidden_output[i][j];
                network.weights_hidden_output[i][j] -= network.learning_rate * self.velocity_weights_hidden_output[i][j];
            }
        }
        
        // Bias salida
        for i in 0..network.bias_output.len() {
            self.velocity_bias_output[i] = self.beta * self.velocity_bias_output[i] + gradients.grad_bias_output[i];
            network.bias_output[i] -= network.learning_rate * self.velocity_bias_output[i];
        }
    }
}

// ================ OPTIMIZADOR RMSPROP ================
/*
RMSPROP EXPLICACIÓN:
- Adapta el learning rate basándose en la magnitud de gradientes recientes
- Mantiene un "cache" de gradientes al cuadrado con decaimiento exponencial
- Fórmula: cache = α * cache_anterior + (1-α) * gradiente²
- Actualización: peso = peso - learning_rate * gradiente / sqrt(cache + ε)
- α típicamente 0.9, ε típicamente 1e-8 (para evitar división por cero)
- Ventaja: se adapta automáticamente a diferentes escalas de gradientes
*/
struct RMSpropOptimizer {
    cache_weights_input_hidden: Vec<Vec<f64>>,
    cache_bias_hidden: Vec<f64>,
    cache_weights_hidden_output: Vec<Vec<f64>>,
    cache_bias_output: Vec<f64>,
    alpha: f64, // Factor de decaimiento
    epsilon: f64, // Para estabilidad numérica
}

impl RMSpropOptimizer {
    fn new(network: &NeuralNetwork, alpha: f64, epsilon: f64) -> Self {
        let cache_weights_input_hidden = vec![vec![0.0; network.weights_input_hidden[0].len()]; network.weights_input_hidden.len()];
        let cache_bias_hidden = vec![0.0; network.bias_hidden.len()];
        let cache_weights_hidden_output = vec![vec![0.0; network.weights_hidden_output[0].len()]; network.weights_hidden_output.len()];
        let cache_bias_output = vec![0.0; network.bias_output.len()];
        
        Self {
            cache_weights_input_hidden,
            cache_bias_hidden,
            cache_weights_hidden_output,
            cache_bias_output,
            alpha,
            epsilon,
        }
    }
    
    fn update(&mut self, network: &mut NeuralNetwork, gradients: &NetworkGradients) {
        // Actualizar cache y pesos entrada → oculta
        for i in 0..network.weights_input_hidden.len() {
            for j in 0..network.weights_input_hidden[i].len() {
                let grad = gradients.grad_weights_input_hidden[i][j];
                self.cache_weights_input_hidden[i][j] = 
                    self.alpha * self.cache_weights_input_hidden[i][j] + (1.0 - self.alpha) * grad * grad;
                let adaptive_lr = network.learning_rate / (self.cache_weights_input_hidden[i][j] + self.epsilon).sqrt();
                network.weights_input_hidden[i][j] -= adaptive_lr * grad;
            }
        }
        
        // Bias oculta
        for i in 0..network.bias_hidden.len() {
            let grad = gradients.grad_bias_hidden[i];
            self.cache_bias_hidden[i] = self.alpha * self.cache_bias_hidden[i] + (1.0 - self.alpha) * grad * grad;
            let adaptive_lr = network.learning_rate / (self.cache_bias_hidden[i] + self.epsilon).sqrt();
            network.bias_hidden[i] -= adaptive_lr * grad;
        }
        
        // Pesos oculta → salida
        for i in 0..network.weights_hidden_output.len() {
            for j in 0..network.weights_hidden_output[i].len() {
                let grad = gradients.grad_weights_hidden_output[i][j];
                self.cache_weights_hidden_output[i][j] = 
                    self.alpha * self.cache_weights_hidden_output[i][j] + (1.0 - self.alpha) * grad * grad;
                let adaptive_lr = network.learning_rate / (self.cache_weights_hidden_output[i][j] + self.epsilon).sqrt();
                network.weights_hidden_output[i][j] -= adaptive_lr * grad;
            }
        }
        
        // Bias salida
        for i in 0..network.bias_output.len() {
            let grad = gradients.grad_bias_output[i];
            self.cache_bias_output[i] = self.alpha * self.cache_bias_output[i] + (1.0 - self.alpha) * grad * grad;
            let adaptive_lr = network.learning_rate / (self.cache_bias_output[i] + self.epsilon).sqrt();
            network.bias_output[i] -= adaptive_lr * grad;
        }
    }
}

// ================ OPTIMIZADOR ADAM ================
/*
ADAM EXPLICACIÓN:
- Combina Momentum + RMSprop
- Mantiene dos momentos: m (momentum) y v (varianza de gradientes)
- m = β1 * m_anterior + (1-β1) * gradiente    (como Momentum)
- v = β2 * v_anterior + (1-β2) * gradiente²   (como RMSprop)
- Corrección de sesgo: m_hat = m / (1 - β1^t), v_hat = v / (1 - β2^t)
- Actualización: peso = peso - learning_rate * m_hat / (sqrt(v_hat) + ε)
- β1 típicamente 0.9, β2 típicamente 0.999, ε típicamente 1e-8
- Ventaja: combina lo mejor de ambos mundos
*/
struct AdamOptimizer {
    m_weights_input_hidden: Vec<Vec<f64>>,
    m_bias_hidden: Vec<f64>,
    m_weights_hidden_output: Vec<Vec<f64>>,
    m_bias_output: Vec<f64>,
    v_weights_input_hidden: Vec<Vec<f64>>,
    v_bias_hidden: Vec<f64>,
    v_weights_hidden_output: Vec<Vec<f64>>,
    v_bias_output: Vec<f64>,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: f64, // Contador de pasos (para corrección de sesgo)
}

impl AdamOptimizer {
    fn new(network: &NeuralNetwork, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        let m_weights_input_hidden = vec![vec![0.0; network.weights_input_hidden[0].len()]; network.weights_input_hidden.len()];
        let m_bias_hidden = vec![0.0; network.bias_hidden.len()];
        let m_weights_hidden_output = vec![vec![0.0; network.weights_hidden_output[0].len()]; network.weights_hidden_output.len()];
        let m_bias_output = vec![0.0; network.bias_output.len()];
        
        let v_weights_input_hidden = vec![vec![0.0; network.weights_input_hidden[0].len()]; network.weights_input_hidden.len()];
        let v_bias_hidden = vec![0.0; network.bias_hidden.len()];
        let v_weights_hidden_output = vec![vec![0.0; network.weights_hidden_output[0].len()]; network.weights_hidden_output.len()];
        let v_bias_output = vec![0.0; network.bias_output.len()];
        
        Self {
            m_weights_input_hidden,
            m_bias_hidden,
            m_weights_hidden_output,
            m_bias_output,
            v_weights_input_hidden,
            v_bias_hidden,
            v_weights_hidden_output,
            v_bias_output,
            beta1,
            beta2,
            epsilon,
            t: 0.0,
        }
    }
    
    fn update(&mut self, network: &mut NeuralNetwork, gradients: &NetworkGradients) {
        self.t += 1.0;
        
        // Factores de corrección de sesgo
        let bias_correction1 = 1.0 - self.beta1.powf(self.t);
        let bias_correction2 = 1.0 - self.beta2.powf(self.t);
        
        // Actualizar momentos y pesos entrada → oculta
        for i in 0..network.weights_input_hidden.len() {
            for j in 0..network.weights_input_hidden[i].len() {
                let grad = gradients.grad_weights_input_hidden[i][j];
                
                // Actualizar momentos
                self.m_weights_input_hidden[i][j] = self.beta1 * self.m_weights_input_hidden[i][j] + (1.0 - self.beta1) * grad;
                self.v_weights_input_hidden[i][j] = self.beta2 * self.v_weights_input_hidden[i][j] + (1.0 - self.beta2) * grad * grad;
                
                // Corrección de sesgo
                let m_hat = self.m_weights_input_hidden[i][j] / bias_correction1;
                let v_hat = self.v_weights_input_hidden[i][j] / bias_correction2;
                
                // Actualizar peso
                network.weights_input_hidden[i][j] -= network.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
            }
        }
        
        // Bias oculta (mismo patrón)
        for i in 0..network.bias_hidden.len() {
            let grad = gradients.grad_bias_hidden[i];
            self.m_bias_hidden[i] = self.beta1 * self.m_bias_hidden[i] + (1.0 - self.beta1) * grad;
            self.v_bias_hidden[i] = self.beta2 * self.v_bias_hidden[i] + (1.0 - self.beta2) * grad * grad;
            let m_hat = self.m_bias_hidden[i] / bias_correction1;
            let v_hat = self.v_bias_hidden[i] / bias_correction2;
            network.bias_hidden[i] -= network.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
        
        // Pesos oculta → salida
        for i in 0..network.weights_hidden_output.len() {
            for j in 0..network.weights_hidden_output[i].len() {
                let grad = gradients.grad_weights_hidden_output[i][j];
                self.m_weights_hidden_output[i][j] = self.beta1 * self.m_weights_hidden_output[i][j] + (1.0 - self.beta1) * grad;
                self.v_weights_hidden_output[i][j] = self.beta2 * self.v_weights_hidden_output[i][j] + (1.0 - self.beta2) * grad * grad;
                let m_hat = self.m_weights_hidden_output[i][j] / bias_correction1;
                let v_hat = self.v_weights_hidden_output[i][j] / bias_correction2;
                network.weights_hidden_output[i][j] -= network.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
            }
        }
        
        // Bias salida
        for i in 0..network.bias_output.len() {
            let grad = gradients.grad_bias_output[i];
            self.m_bias_output[i] = self.beta1 * self.m_bias_output[i] + (1.0 - self.beta1) * grad;
            self.v_bias_output[i] = self.beta2 * self.v_bias_output[i] + (1.0 - self.beta2) * grad * grad;
            let m_hat = self.m_bias_output[i] / bias_correction1;
            let v_hat = self.v_bias_output[i] / bias_correction2;
            network.bias_output[i] -= network.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}

// ================ ESTRUCTURA PARA RESULTADOS ================
#[derive(Debug)]
struct OptimizerResult {
    name: String,
    final_error: f64,
    epochs_completed: usize,
    converged: bool,
    training_time_ms: u128,
}

// ================ FUNCIÓN DE ENTRENAMIENTO ================
fn train_with_optimizer<T>(
    mut network: NeuralNetwork,
    mut optimizer: T,
    optimizer_name: String,
    training_data: &[([f64; 2], [f64; 1])],
    max_epochs: usize,
    target_error: f64,
    progress_sender: std::sync::mpsc::Sender<String>,
) -> OptimizerResult
where
    T: OptimizerTrait,
{
    let start_time = Instant::now();
    let mut converged = false;
    let mut epochs_completed = 0;
    let mut final_error = f64::INFINITY;
    
    for epoch in 0..max_epochs {
        let mut total_error = 0.0;
        
        // Entrenar con todos los ejemplos
        for (inputs, targets) in training_data {
            let (gradients, error) = network.compute_gradients(inputs, targets);
            optimizer.update(&mut network, &gradients);
            total_error += error;
        }
        
        final_error = total_error;
        epochs_completed = epoch + 1;
        
        // Mostrar progreso cada 500 épocas
        if epoch % 500 == 0 {
            let msg = format!("{}: Época {}, Error = {:.6}", optimizer_name, epoch, total_error);
            let _ = progress_sender.send(msg);
        }
        
        // Verificar convergencia
        if total_error < target_error {
            converged = true;
            let msg = format!("🎉 {} CONVERGIÓ en época {} con error {:.6}!", optimizer_name, epoch, total_error);
            let _ = progress_sender.send(msg);
            break;
        }
    }
    
    let training_time_ms = start_time.elapsed().as_millis();
    
    OptimizerResult {
        name: optimizer_name,
        final_error,
        epochs_completed,
        converged,
        training_time_ms,
    }
}

// ================ TRAIT PARA POLIMORFISMO ================
trait OptimizerTrait {
    fn update(&mut self, network: &mut NeuralNetwork, gradients: &NetworkGradients);
}

impl OptimizerTrait for MomentumOptimizer {
    fn update(&mut self, network: &mut NeuralNetwork, gradients: &NetworkGradients) {
        self.update(network, gradients);
    }
}

impl OptimizerTrait for RMSpropOptimizer {
    fn update(&mut self, network: &mut NeuralNetwork, gradients: &NetworkGradients) {
        self.update(network, gradients);
    }
}

impl OptimizerTrait for AdamOptimizer {
    fn update(&mut self, network: &mut NeuralNetwork, gradients: &NetworkGradients) {
        self.update(network, gradients);
    }
}

// ================ FUNCIÓN PRINCIPAL ================
fn main() {
    println!("🧠 COMPARACIÓN DE OPTIMIZADORES: Momentum vs RMSprop vs Adam");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // Datos de entrenamiento: función XOR
    let training_data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];
    
    println!("📊 Datos de entrenamiento (función XOR):");
    for (inputs, targets) in &training_data {
        println!("   {:?} → {:?}", inputs, targets);
    }
    println!();
    
    // Parámetros de entrenamiento
    let max_epochs = 10000;
    let target_error = 0.01; // Error objetivo para convergencia
    let learning_rate = 0.1;
    let seed = 42; // Misma semilla para todos los optimizadores
    
    // Crear canal para comunicación entre hilos
    let (progress_sender, progress_receiver) = std::sync::mpsc::channel::<String>();
    let results = Arc::new(Mutex::new(Vec::new()));
    
    println!("🚀 Iniciando entrenamiento en paralelo...");
    println!("   Learning rate: {}", learning_rate);
    println!("   Épocas máximas: {}", max_epochs);
    println!("   Error objetivo: {}", target_error);
    println!();
    
    // ================ HILO 1: MOMENTUM ================
    let progress_sender1 = progress_sender.clone();
    let results1 = Arc::clone(&results);
    let handle1 = thread::spawn(move || {
        let network = NeuralNetwork::new_with_seed(2, 4, 1, learning_rate, seed);
        let optimizer = MomentumOptimizer::new(&network, 0.9); // β = 0.9
        
        let result = train_with_optimizer(
            network,
            optimizer,
            "MOMENTUM".to_string(),
            &training_data,
            max_epochs,
            target_error,
            progress_sender1,
        );
        
        results1.lock().unwrap().push(result);
    });
    
    // ================ HILO 2: RMSPROP ================
    let progress_sender2 = progress_sender.clone();
    let results2 = Arc::clone(&results);
    let handle2 = thread::spawn(move || {
        let network = NeuralNetwork::new_with_seed(2, 4, 1, learning_rate, seed);
        let optimizer = RMSpropOptimizer::new(&network, 0.9, 1e-8); // α = 0.9, ε = 1e-8
        
        let result = train_with_optimizer(
            network,
            optimizer,
            "RMSPROP".to_string(),
            &training_data,
            max_epochs,
            target_error,
            progress_sender2,
        );
        
        results2.lock().unwrap().push(result);
    });
    
    // ================ HILO 3: ADAM ================
    let progress_sender3 = progress_sender.clone();
        let results3 = Arc::clone(&results);
    let handle3 = thread::spawn(move || {
        let network = NeuralNetwork::new_with_seed(2, 4, 1, learning_rate, seed);
        let optimizer = AdamOptimizer::new(&network, 0.9, 0.999, 1e-8); // β1 = 0.9, β2 = 0.999, ε = 1e-8

        let result = train_with_optimizer(
            network,
            optimizer,
            "ADAM".to_string(),
            &training_data,
            max_epochs,
            target_error,
            progress_sender3,
        );

        results3.lock().unwrap().push(result);
    });

    // Mostrar progreso en tiempo real desde cualquier hilo
    let progress_handle = thread::spawn(move || {
        for message in progress_receiver {
            println!("{}", message);
        }
    });

    // Esperar a que terminen los hilos de entrenamiento
    handle1.join().unwrap();
    handle2.join().unwrap();
    handle3.join().unwrap();

    // Esperar a que se impriman todos los mensajes
    drop(progress_sender); // Cerramos el canal
    progress_handle.join().unwrap();

    // Mostrar resultados finales
    let final_results = results.lock().unwrap();
    println!("\n📈 RESULTADOS FINALES");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    for res in final_results.iter() {
        println!(
            "🔹 {} → Error final: {:.6} | Épocas: {} | Tiempo: {}ms {}",
            res.name,
            res.final_error,
            res.epochs_completed,
            res.training_time_ms,
            if res.converged { "✅" } else { "❌" }
        );
    }

    // Mostrar el ganador
    if let Some(best) = final_results.iter().min_by(|a, b| a.final_error.partial_cmp(&b.final_error).unwrap()) {
        println!(
            "\n🏆 GANADOR: {} con error final {:.6} en {} épocas ({}ms)",
            best.name, best.final_error, best.epochs_completed, best.training_time_ms
        );
    }
}
