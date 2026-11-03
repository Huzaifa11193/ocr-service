from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    service_token: str = "change-me-long-random"
    ollama_url: str = "http://ollama:11434"
    ollama_vision_model: str = "qwen2.5vl:7b"
    tesseract_lang: str = "urd+eng"
    ocr_dpi: int = 300
    ocr_workers: int = 2
    confidence_threshold: int = 60

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
