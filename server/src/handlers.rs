use axum::{extract::State, Json};
use shared::{TtsRequest, TtsResponse, HealthResponse};
use std::sync::Arc;
use crate::model::KannadaTtsModel;

pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
    })
}

pub async fn synthesize(
    State(model): State<Arc<KannadaTtsModel>>,
    Json(request): Json<TtsRequest>,
) -> Json<TtsResponse> {
    // Normalize and tokenize text
    let tokens = match crate::tokenizer::tokenize(&request.text) {
        Ok(t) => t,
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
    match model.synthesize(&tokens) {
        Ok(waveform) => {
            let sample_rate = model.get_sample_rate();
            Json(TtsResponse {
                audio_data: waveform,
                sample_rate,
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
