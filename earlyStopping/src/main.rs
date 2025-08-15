use plotters::prelude::*;

// ================ TRAITS PARA REGULARIZACIÓN ================
trait Regularizer {
    fn compute_loss(&self, network: &NeuralNetwork) -> f64;
    fn apply_to_gradient(&self, gradient: f64, weight: f64) -> f64;
}

// ================ REGULARIZADOR L1 ================
#[derive(Debug, Clone)]
struct L1Regularizer {
    lambda: f64,
}

impl L1Regularizer {
    fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

impl Regularizer for L1Regularizer {
    fn compute_loss(&self, network: &NeuralNetwork) -> f64 {
        let mut l1_loss = 0.0;
        
        // L1 de pesos entrada → oculta
        for row in &network.weights_input_hidden {
            for &weight in row {
                l1_loss += self.lambda * weight.abs();
            }
        }
        
        // L1 de pesos oculta → salida
        for row in &network.weights_hidden_output {
            for &weight in row {
                l1_loss += self.lambda * weight.abs();
            }
        }
        
        l1_loss
    }
    
    fn apply_to_gradient(&self, gradient: f64, weight: f64) -> f64 {
        // L1: añade λ * sign(weight) al gradiente
        gradient + self.lambda * weight.signum()
    }
    
    
}

// ================ REGULARIZADOR L2 ================
#[derive(Debug, Clone)]
struct L2Regularizer {
    lambda: f64,
}

impl L2Regularizer {
    fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

impl Regularizer for L2Regularizer {
    fn compute_loss(&self, network: &NeuralNetwork) -> f64 {
        let mut l2_loss = 0.0;
        
        // L2 de pesos entrada → oculta
        for row in &network.weights_input_hidden {
            for &weight in row {
                l2_loss += 0.5 * self.lambda * weight * weight ;
            }
        }
        
        // L2 de pesos oculta → salida
        for row in &network.weights_hidden_output {
            for &weight in row {
                l2_loss += 0.5 * self.lambda * weight * weight;
            }
        }
        
        l2_loss
    }
    
    fn apply_to_gradient(&self, gradient: f64, weight: f64) -> f64 {
        // L2: añade λ * weight al gradiente
        gradient + self.lambda * weight
    }
    
   
}

// ================ REGULARIZADOR DROPOUT ================
#[derive(Debug, Clone)]
struct DropoutRegularizer {
    rate: f64,
}

impl DropoutRegularizer {
    fn new(rate: f64) -> Self {
        Self { rate }
    }
    
    // Dropout no contribuye a la función de pérdida
    fn apply_dropout(&self, hidden: &mut Vec<f64>, rng_state: &mut u64, training: bool) -> Vec<bool> {
        let mut mask = vec![true; hidden.len()];
        
        if !training {
            return mask; // Sin dropout durante evaluación
        }
        
        for i in 0..hidden.len() {
            // Generar número aleatorio
            *rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let random_val = ((*rng_state / 65536) % 32768) as f64 / 32768.0;
            
            if random_val < self.rate {
                hidden[i] = 0.0; // Apagar neurona
                mask[i] = false;
            } else {
                // Escalar por (1 - dropout_rate) para mantener la escala esperada
                hidden[i] /= 1.0 - self.rate;
            }
        }
        
        mask
    }
}

impl Regularizer for DropoutRegularizer {
    fn compute_loss(&self, _network: &NeuralNetwork) -> f64 {
        0.0 // Dropout no añade pérdida explícita
    }
    
    fn apply_to_gradient(&self, gradient: f64, _weight: f64) -> f64 {
        gradient // Dropout no modifica gradientes directamente
    }
    
    
}

// ================ RED NEURONAL MODULAR ================
#[derive(Debug, Clone)]
struct NeuralNetwork {
    weights_input_hidden: Vec<Vec<f64>>,
    bias_hidden: Vec<f64>,
    weights_hidden_output: Vec<Vec<f64>>,
    bias_output: Vec<f64>,
    learning_rate: f64,
    dropout: Option<DropoutRegularizer>,
    rng_state: u64,
}

impl NeuralNetwork {
    fn new_with_seed(
        input_size: usize, 
        hidden_size: usize, 
        output_size: usize, 
        learning_rate: f64, 
        seed: u64
    ) -> Self {
        let mut rng_state = seed;
        
        let mut next_random = || {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            (rng_state / 65536) % 32768
        };
        
        let mut random_weight = || ((next_random() as f64) / 32768.0) * 2.0 - 1.0;
        
        // Inicializar pesos entrada → oculta
        let mut weights_input_hidden = Vec::new();
        for _i in 0..input_size {
            let mut row = Vec::new();
            for _j in 0..hidden_size {
                row.push(random_weight() * 0.5);
            }
            weights_input_hidden.push(row);
        }
        
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
            dropout: None,
            rng_state,
        }
    }
    
    // Configurar dropout
    fn with_dropout(mut self, rate: f64) -> Self {
        self.dropout = Some(DropoutRegularizer::new(rate));
        self
    }
    
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    
    // ================ FORWARD PASS ================
    fn forward(&mut self, inputs: &[f64], training: bool) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
        // Capa oculta
        let mut hidden = Vec::new();
        for h in 0..self.bias_hidden.len() {
            let mut sum = self.bias_hidden[h];
            for (i, &input) in inputs.iter().enumerate() {
                sum += input * self.weights_input_hidden[i][h];
            }
            hidden.push(Self::sigmoid(sum));
        }
        
        // Aplicar dropout si está configurado
        let dropout_mask = if let Some(ref dropout) = self.dropout {
            dropout.apply_dropout(&mut hidden, &mut self.rng_state, training)
        } else {
            vec![true; hidden.len()]
        };
        
        // Capa de salida
        let mut output = Vec::new();
        for o in 0..self.bias_output.len() {
            let mut sum = self.bias_output[o];
            for (h, &hidden_val) in hidden.iter().enumerate() {
                sum += hidden_val * self.weights_hidden_output[h][o];
            }
            output.push(Self::sigmoid(sum));
        }
        
        (hidden, output, dropout_mask)
    }
    
    // ================ GRADIENTES CON REGULARIZADOR OPCIONAL ================
    fn compute_gradients<R: Regularizer>(&mut self, inputs: &[f64], targets: &[f64], regularizer: Option<&R>) -> (NetworkGradients, f64) {
        let (hidden, output, dropout_mask) = self.forward(inputs, true);
        
        // Error base
        let mut output_errors = Vec::new();
        let mut base_error = 0.0;
        
        for (&out, &target) in output.iter().zip(targets.iter()) {
            let error = target - out;
            base_error += 0.5 * error.powi(2);
            let delta = -error * out * (1.0 - out);
            output_errors.push(delta);
        }
        
        // Añadir pérdida de regularización si hay regularizador
        let reg_loss = if let Some(reg) = regularizer {
            reg.compute_loss(self)
        } else {
            0.0
        };
        let total_error = base_error + reg_loss;
        
        // Errores de capa oculta (considerando dropout)
        let mut hidden_errors = Vec::new();
        for h in 0..hidden.len() {
            let mut error_sum = 0.0;
            for (o, &output_error) in output_errors.iter().enumerate() {
                error_sum += output_error * self.weights_hidden_output[h][o];
            }
            
            let delta = if dropout_mask[h] {
                error_sum * hidden[h] * (1.0 - hidden[h])
            } else {
                0.0
            };
            hidden_errors.push(delta);
        }
        
        // ================ GRADIENTES CON REGULARIZACIÓN ================
        let mut grad_weights_hidden_output = vec![vec![0.0; self.weights_hidden_output[0].len()]; self.weights_hidden_output.len()];
        for h in 0..hidden.len() {
            for (o, &output_error) in output_errors.iter().enumerate() {
                let base_grad = output_error * hidden[h];
                let final_grad = if let Some(reg) = regularizer {
                    reg.apply_to_gradient(base_grad, self.weights_hidden_output[h][o])
                } else {
                    base_grad
                };
                grad_weights_hidden_output[h][o] = final_grad;
            }
        }
        
        let mut grad_weights_input_hidden = vec![vec![0.0; self.weights_input_hidden[0].len()]; self.weights_input_hidden.len()];
        for (i, &input) in inputs.iter().enumerate() {
            for (h, &hidden_error) in hidden_errors.iter().enumerate() {
                let base_grad = hidden_error * input;
                let final_grad = if let Some(reg) = regularizer {
                    reg.apply_to_gradient(base_grad, self.weights_input_hidden[i][h])
                } else {
                    base_grad
                };
                grad_weights_input_hidden[i][h] = final_grad;
            }
        }
        
        let gradients = NetworkGradients {
            grad_weights_input_hidden,
            grad_bias_hidden: hidden_errors.clone(),
            grad_weights_hidden_output,
            grad_bias_output: output_errors.clone(),
        };
        
        (gradients, total_error)
    }
    
    // Evaluación sin dropout
    fn evaluate(&mut self, inputs: &[f64]) -> Vec<f64> {
        let (_, output, _) = self.forward(inputs, false);
        output
    }
}

// ================ ESTRUCTURAS AUXILIARES ================
#[derive(Debug, Clone)]
struct NetworkGradients {
    grad_weights_input_hidden: Vec<Vec<f64>>,
    grad_bias_hidden: Vec<f64>,
    grad_weights_hidden_output: Vec<Vec<f64>>,
    grad_bias_output: Vec<f64>,
}

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
    t: f64,
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
            m_weights_input_hidden, m_bias_hidden, m_weights_hidden_output, m_bias_output,
            v_weights_input_hidden, v_bias_hidden, v_weights_hidden_output, v_bias_output,
            beta1, beta2, epsilon, t: 0.0,
        }
    }
    
    fn update(&mut self, network: &mut NeuralNetwork, gradients: &NetworkGradients) {
        self.t += 1.0;
        let bias_correction1 = 1.0 - self.beta1.powf(self.t);
        let bias_correction2 = 1.0 - self.beta2.powf(self.t);
        
        // Actualizar pesos entrada → oculta
        for i in 0..network.weights_input_hidden.len() {
            for j in 0..network.weights_input_hidden[i].len() {
                let grad = gradients.grad_weights_input_hidden[i][j];
                self.m_weights_input_hidden[i][j] = self.beta1 * self.m_weights_input_hidden[i][j] + (1.0 - self.beta1) * grad;
                self.v_weights_input_hidden[i][j] = self.beta2 * self.v_weights_input_hidden[i][j] + (1.0 - self.beta2) * grad * grad;
                let m_hat = self.m_weights_input_hidden[i][j] / bias_correction1;
                let v_hat = self.v_weights_input_hidden[i][j] / bias_correction2;
                network.weights_input_hidden[i][j] -= network.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
            }
        }
        
        // Bias oculta
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

// ================ FUNCIÓN DE ENTRENAMIENTO GENÉRICA ================
fn train_network<R: Regularizer>(
    mut network: NeuralNetwork,
    mut optimizer: AdamOptimizer,
    training_data: &[([f64; 2], [f64; 1])],
    validation_data: &[([f64; 2], [f64; 1])],
    max_epochs: usize,
    target_error: f64,
    regularizer: Option<R>,
    name: &str,
) -> (f64, f64, bool, usize) {
    println!("\n🔹 Entrenando con: {}", name);
    
    let mut converged = false;
    let mut epochs_completed = 0;
    
    for epoch in 0..max_epochs {
        // Entrenamiento
        let mut train_error = 0.0;
        for (inputs, targets) in training_data {
            let (gradients, error) = network.compute_gradients(inputs, targets, regularizer.as_ref());
            optimizer.update(&mut network, &gradients);
            train_error += error;
        }
        
        // Validación (sin regularización en el error mostrado)
        let mut val_error = 0.0;
        for (inputs, targets) in validation_data {
            let output = network.evaluate(inputs);
            for (&out, &target) in output.iter().zip(targets.iter()) {
                let error = target - out;
                val_error += 0.5 * error.powi(2);
            }
        }
        
        epochs_completed = epoch + 1;
        
        // Mostrar progreso cada 500 épocas
        if epoch % 500 == 0 {
            println!("  {}: Época {}, Error Val = {:.6}", name, epoch, val_error);
        }
        
        // Verificar convergencia usando error de validación
        if val_error < target_error {
            converged = true;
            println!("🎉 {} CONVERGIÓ en época {} con error {:.6}!", name, epoch, val_error);
            break;
        }
    }
    
    // Error final
    let mut final_train = 0.0;
    let mut final_val = 0.0;
    
    for (inputs, targets) in training_data {
        let output = network.evaluate(inputs);
        for (&out, &target) in output.iter().zip(targets.iter()) {
            let error = target - out;
            final_train += 0.5 * error.powi(2);
        }
    }
    
    for (inputs, targets) in validation_data {
        let output = network.evaluate(inputs);
        for (&out, &target) in output.iter().zip(targets.iter()) {
            let error = target - out;
            final_val += 0.5 * error.powi(2);
        }
    }
    
    // Mostrar resultado final
    let status = if converged { "✅ APRENDIÓ" } else { "❌ NO CONVERGIÓ" };
    println!("  → {} en {} épocas: Train={:.6}, Val={:.6}", status, epochs_completed, final_train, final_val);
    
    (final_train, final_val, converged, epochs_completed)
}

// ================ EARLY STOPPING IMPLEMENTATION ================

#[derive(Debug, Clone)]
struct EarlyStopping {
    patience: usize,           // Épocas a esperar sin mejora
    min_delta: f64,           // Mínima mejora considerada significativa
    wait_counter: usize,      // Contador de épocas sin mejora
    best_loss: f64,          // Mejor pérdida de validación encontrada
    best_weights: Option<NetworkWeights>, // Mejores pesos guardados
    improved_at_epoch: usize, // Época donde se encontró la mejor pérdida
}

#[derive(Debug, Clone)]
struct NetworkWeights {
    weights_input_hidden: Vec<Vec<f64>>,
    bias_hidden: Vec<f64>,
    weights_hidden_output: Vec<Vec<f64>>,
    bias_output: Vec<f64>,
}

impl EarlyStopping {
    /// Crear nuevo Early Stopping
    /// - patience: número de épocas a esperar sin mejora antes de parar
    /// - min_delta: mínima mejora requerida (ej: 0.001 significa que necesita mejorar al menos 0.1%)
    fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            wait_counter: 0,
            best_loss: f64::INFINITY,
            best_weights: None,
            improved_at_epoch: 0,
        }
    }
    
    /// Evaluar si debe continuar el entrenamiento
    /// Retorna: (should_continue, improved_this_epoch)
    fn should_continue(&mut self, current_val_loss: f64, current_epoch: usize, network: &NeuralNetwork) -> (bool, bool) {
        let improvement = self.best_loss - current_val_loss;
        let improved = improvement > self.min_delta;
        
        if improved {
            // Mejora significativa detectada
            self.best_loss = current_val_loss;
            self.wait_counter = 0;
            self.improved_at_epoch = current_epoch;
            
            // Guardar los mejores pesos
            self.best_weights = Some(NetworkWeights {
                weights_input_hidden: network.weights_input_hidden.clone(),
                bias_hidden: network.bias_hidden.clone(),
                weights_hidden_output: network.weights_hidden_output.clone(),
                bias_output: network.bias_output.clone(),
            });
            
            (true, true)
        } else {
            // No hay mejora significativa
            self.wait_counter += 1;
            
            if self.wait_counter >= self.patience {
                (false, false) // Parar entrenamiento
            } else {
                (true, false)  // Continuar entrenamiento
            }
        }
    }
    
    /// Restaurar los mejores pesos encontrados
    fn restore_best_weights(&self, network: &mut NeuralNetwork) -> bool {
        if let Some(ref best_weights) = self.best_weights {
            network.weights_input_hidden = best_weights.weights_input_hidden.clone();
            network.bias_hidden = best_weights.bias_hidden.clone();
            network.weights_hidden_output = best_weights.weights_hidden_output.clone();
            network.bias_output = best_weights.bias_output.clone();
            true
        } else {
            false
        }
    }
    
    /// Obtener información del estado actual
    fn get_info(&self) -> (f64, usize, usize, usize) {
        (self.best_loss, self.wait_counter, self.patience, self.improved_at_epoch)
    }
}

// ================ FUNCIÓN DE ENTRENAMIENTO CON EARLY STOPPING ================
fn train_network_with_early_stopping<R: Regularizer>(
    mut network: NeuralNetwork,
    mut optimizer: AdamOptimizer,
    training_data: &[([f64; 2], [f64; 1])],
    validation_data: &[([f64; 2], [f64; 1])],
    max_epochs: usize,
    regularizer: Option<R>,
    early_stopping: Option<EarlyStopping>,
    name: &str,
) -> (f64, f64, bool, usize, bool) {
    println!("\n🔹 Entrenando con Early Stopping: {}", name);
    
    let mut early_stop = early_stopping.unwrap_or_else(|| EarlyStopping::new(100, 0.0001));
    let mut epochs_completed = 0;
    let mut stopped_early = false;

    let mut train_errors: Vec<f64> = Vec::new();
    let mut val_errors: Vec<f64> = Vec::new();
    
    for epoch in 0..max_epochs {
        // Entrenamiento
        let mut train_error = 0.0;
        for (inputs, targets) in training_data {
            let (gradients, error) = network.compute_gradients(inputs, targets, regularizer.as_ref());
            optimizer.update(&mut network, &gradients);
            train_error += error;
        }
        
        // Validación (sin regularización en el error mostrado)
        let mut val_error = 0.0;
        for (inputs, targets) in validation_data {
            let output = network.evaluate(inputs);
            for (&out, &target) in output.iter().zip(targets.iter()) {
                let error = target - out;
                val_error += 0.5 * error.powi(2);
            }
        }
        
        epochs_completed = epoch + 1;
        
        // Evaluar Early Stopping
        let (should_continue, improved) = early_stop.should_continue(val_error, epoch, &network);
        let (best_loss, wait_counter, patience, best_epoch) = early_stop.get_info();
        
        train_errors.push(train_error);
        val_errors.push(val_error);
        // Mostrar progreso cada 500 épocas o cuando hay mejora
        if epoch % 500 == 0 || improved {
            let status = if improved { "🎯 MEJORA" } else { "📊" };
            println!("  {} Época {}: Val={:.6}, Mejor={:.6} (época {}), Espera={}/{}", 
                     status, epoch, val_error, best_loss, best_epoch, wait_counter, patience);
        }
        
        if !should_continue {
            stopped_early = true;
            println!("🛑 Early Stopping activado en época {} (sin mejora por {} épocas)", 
                     epoch, patience);
            println!("   Mejor validación: {:.6} en época {}", best_loss, best_epoch);
            break;
        }
    }
    
    // Restaurar mejores pesos
    let restored = early_stop.restore_best_weights(&mut network);
    if restored {
        println!("🔄 Pesos restaurados al mejor modelo (época {})", early_stop.improved_at_epoch);
    }
    
    // Error final con los mejores pesos
    let mut final_train = 0.0;
    let mut final_val = 0.0;
    
    for (inputs, targets) in training_data {
        let output = network.evaluate(inputs);
        for (&out, &target) in output.iter().zip(targets.iter()) {
            let error = target - out;
            final_train += 0.5 * error.powi(2);
        }
    }
    
    for (inputs, targets) in validation_data {
        let output = network.evaluate(inputs);
        for (&out, &target) in output.iter().zip(targets.iter()) {
            let error = target - out;
            final_val += 0.5 * error.powi(2);
        }
    }
    
    // Mostrar resultado final
    let early_status = if stopped_early { "🛑 PARADA TEMPRANA" } else { "📈 ÉPOCAS COMPLETAS" };
    let restore_status = if restored { " (pesos restaurados)" } else { "" };
    
    if let Err(err) = plot_training_curve(&train_errors, &val_errors, "training_curve.png") {
        eprintln!("Error al generar la gráfica: {}", err);
    }
    println!("  → {} en {} épocas: Train={:.6}, Val={:.6}{}", 
             early_status, epochs_completed, final_train, final_val, restore_status);
    
    (final_train, final_val, true, epochs_completed, stopped_early)
}

fn plot_training_curve(train_errors: &[f64], val_errors: &[f64], filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_epoch = train_errors.len();
    let max_val = train_errors
        .iter()
        .chain(val_errors.iter())
        .cloned()
        .fold(f64::MIN, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Curva de Entrenamiento", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0..max_epoch, 0f64..max_val)?;

    chart.configure_mesh()
        .x_desc("Épocas")
        .y_desc("Error")
        .draw()?;

    // Línea de entrenamiento
    chart.draw_series(LineSeries::new(
        train_errors.iter().enumerate().map(|(i, &e)| (i, e)),
        &RED,
    ))?.label("Entrenamiento").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Línea de validación
    chart.draw_series(LineSeries::new(
        val_errors.iter().enumerate().map(|(i, &e)| (i, e)),
        &BLUE,
    ))?.label("Validación").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE)
        .draw()?;

    Ok(())
}

// ================ FUNCIÓN PRINCIPAL ================
fn main() {
    println!("🧠 RED NEURONAL CON EARLY STOPPING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let training_data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.1, 1.0], [0.0]),
    ];
    
    let validation_data = training_data.clone();
    let max_epochs = 10000;
    
    println!("📊 Datos de entrenamiento (función XOR):");
    for (inputs, targets) in &training_data {
        println!("   {:?} → {:?}", inputs, targets);
    }
    println!("🎯 Épocas máximas: {}\n", max_epochs);
    
    let mut results = Vec::new();
    
    // 1. Sin Early Stopping (para comparación)
    {
        let network = NeuralNetwork::new_with_seed(2, 8, 1, 0.1, 42);
        let optimizer = AdamOptimizer::new(&network, 0.9, 0.999, 1e-8);
        let (train_err, val_err, _, epochs, early_stopped) = train_network_with_early_stopping(
            network, optimizer, &training_data, &validation_data, 2000,
            None::<L1Regularizer>, None, "Sin Early Stopping (2000 épocas)"
        );
        results.push(("Sin Early Stopping", train_err, val_err, epochs, early_stopped));
    }
    
    // 2. Con Early Stopping conservador
    {
        let network = NeuralNetwork::new_with_seed(2, 8, 1, 0.1, 42);
        let optimizer = AdamOptimizer::new(&network, 0.9, 0.999, 1e-8);
        let early_stop = EarlyStopping::new(200, 0.001); // Patience=200, min_delta=0.001
        let (train_err, val_err, _, epochs, early_stopped) = train_network_with_early_stopping(
            network, optimizer, &training_data, &validation_data, max_epochs,
            None::<L1Regularizer>, Some(early_stop), "Early Stopping Conservador"
        );
        results.push(("Early Stop Conservador", train_err, val_err, epochs, early_stopped));
    }
    
    // 3. Con Early Stopping agresivo
    {
        let network = NeuralNetwork::new_with_seed(2, 8, 1, 0.1, 42);
        let optimizer = AdamOptimizer::new(&network, 0.9, 0.999, 1e-8);
        let early_stop = EarlyStopping::new(50, 0.0001); // Patience=50, min_delta=0.0001
        let (train_err, val_err, _, epochs, early_stopped) = train_network_with_early_stopping(
            network, optimizer, &training_data, &validation_data, max_epochs,
            None::<L1Regularizer>, Some(early_stop), "Early Stopping Agresivo"
        );
        results.push(("Early Stop Agresivo", train_err, val_err, epochs, early_stopped));
    }
    
    // 4. Early Stopping + L2 Regularización
    {
        let network = NeuralNetwork::new_with_seed(2, 8, 1, 0.1, 42);
        let optimizer = AdamOptimizer::new(&network, 0.9, 0.999, 1e-8);
        let l2_reg = L2Regularizer::new(0.0001);
        let early_stop = EarlyStopping::new(100, 0.0005);
        let (train_err, val_err, _, epochs, early_stopped) = train_network_with_early_stopping(
            network, optimizer, &training_data, &validation_data, max_epochs,
            Some(l2_reg), Some(early_stop), "Early Stopping + L2"
        );
        results.push(("Early Stop + L2", train_err, val_err, epochs, early_stopped));
    }
    
    // 5. Early Stopping + Dropout
    {
        let network = NeuralNetwork::new_with_seed(2, 8, 1, 0.1, 42).with_dropout(0.3);
        let optimizer = AdamOptimizer::new(&network, 0.9, 0.999, 1e-8);
        let early_stop = EarlyStopping::new(150, 0.0005);
        let (train_err, val_err, _, epochs, early_stopped) = train_network_with_early_stopping(
            network, optimizer, &training_data, &validation_data, max_epochs,
            None::<L1Regularizer>, Some(early_stop), "Early Stopping + Dropout"
        );
        results.push(("Early Stop + Dropout", train_err, val_err, epochs, early_stopped));
    }
    
    // Resumen final
    println!("\n📈 RESUMEN DE EARLY STOPPING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    for (name, train_err, val_err, epochs, early_stopped) in &results {
        let stop_indicator = if *early_stopped { "🛑" } else { "📈" };
        println!("🔹 {} {} → Épocas: {}, Train: {:.6}, Val: {:.6}", 
                 name, stop_indicator, epochs, train_err, val_err);
    }
    
    // Mostrar el más eficiente (mejor error en menos épocas)
    if let Some(best) = results.iter()
        .min_by(|(_, _, val_err1, epochs1, _), (_, _, val_err2, epochs2, _)| {
            // Priorizar error bajo, luego menos épocas
            val_err1.partial_cmp(val_err2).unwrap_or(epochs1.cmp(epochs2))
        }) {
        println!("\n🏆 MÁS EFICIENTE: {} con error {:.6} en {} épocas", 
                 best.0, best.2, best.3);
        
        if best.4 {
            println!("   ✨ Se benefició del Early Stopping");
        }
    }
    
    println!("\n✅ Demostración de Early Stopping completada!");
    println!("💡 Early Stopping ayuda a:");
    println!("   • Evitar sobreajuste (overfitting)");
    println!("   • Reducir tiempo de entrenamiento");
    println!("   • Encontrar el modelo con mejor generalización");
}
