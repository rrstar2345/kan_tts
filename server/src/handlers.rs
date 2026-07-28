use axum::{extract::State, Json};
use shared::{TtsRequest, TtsResponse, HealthResponse};
use std::sync::Arc;
use crate::state::AppState;

pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
    })
}

pub async fn synthesize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TtsRequest>,
) -> Json<TtsResponse> {
    // Normalize and tokenize text
    let tokens = match state.tokenizer.tokenize(&request.text) {
        Ok(tokens) => tokens,
        Err(e) => {
            return Json(TtsResponse {
                audio_data: vec![],
                sample_rate: 0,
                success: false,
                message: format!("Tokenization error: {}", e),
            });
        }
    };

    // Run inference
    let mut model = state.model.lock().await;

    match model.synthesize(&tokens) {
        Ok(waveform) => {
            Json(TtsResponse {
                audio_data: waveform,
                sample_rate: model.get_sample_rate(),
                success: true,
                message: "Synthesis successful".to_string(),
            })
        }
        Err(e) => {
            tracing::error!("Synthesis error: {}", e);

            Json(TtsResponse {
                audio_data: vec![],
                sample_rate: 0,
                success: false,
                message: format!("Synthesis error: {}", e),
            })
        }
    }
}
