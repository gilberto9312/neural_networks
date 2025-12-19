// Common LLM Library
// Código compartido para proyectos LLM (días 15-21)

pub mod activation;
pub mod loss;
pub mod optimizer;
pub mod dataset_loader;
pub mod metrics;

// Re-exportar estructuras comunes
pub use activation::*;
pub use loss::*;
pub use optimizer::*;
pub use dataset_loader::*;
pub use metrics::*;
