use tokio::sync::Mutex;

use crate::{
    model::KannadaTtsModel,
    tokenizer::KannadaTokenizer,
};

pub struct AppState {
    pub model: Mutex<KannadaTtsModel>,
    pub tokenizer: KannadaTokenizer,
}