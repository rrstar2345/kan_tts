mod handlers;
mod model;
mod tokenizer;
mod state;

use axum::{
    // extract::State,
    // response::Json,
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use state::AppState;
use tokio::sync::Mutex;
use tracing_subscriber;

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Load model
    let model = match model::KannadaTtsModel::new("./models/model.onnx") {
        Ok(m) => {
            tracing::info!("✓ Model loaded successfully");
            m
        }
        Err(e) => {
            tracing::error!("✗ Failed to load model: {}", e);
            std::process::exit(1);
        }
    };

    let tokenizer = match tokenizer::KannadaTokenizer::new(
        "./models/tokenizer.json",
    ) {
        Ok(t) => {
            tracing::info!("✓ Tokenizer loaded successfully");
            t
        }
        Err(e) => {
            tracing::error!("✗ Failed to load tokenizer: {}", e);
            std::process::exit(1);
        }
    };

    let state = Arc::new(AppState {
        model: Mutex::new(model),
        tokenizer,
    });

    // Build router
    let app = Router::new()
        .route("/health", get(handlers::health))
        .route("/synthesize", post(handlers::synthesize))
        .with_state(state);

    // Start server
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080")
        .await
        .expect("Failed to bind to port 8080");

    tracing::info!("🎙️  Kannada TTS Server running on http://127.0.0.1:8080");

    axum::serve(listener, app)
        .await
        .expect("Server error");
}
