use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TtsRequest {
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TtsResponse {
    pub audio_data: Vec<f32>,
    pub sample_rate: u32,
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
}
