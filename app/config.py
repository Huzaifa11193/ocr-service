from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    service_token: str = "change-me-long-random"
    ollama_url: str = "http://ollama:11434"
    ollama_vision_model: str = "qwen2.5vl:7b"
    tesseract_lang: str = "urd+eng"
    ocr_dpi: int = 300
    ocr_workers: int = 2
    confidence_threshold: int = 60
    # Audio transcription (POST /transcribe); tiny|base|small, int8 on CPU.
    whisper_model: str = "base"
    whisper_compute_type: str = "int8"
    max_audio_mb: int = 25

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
