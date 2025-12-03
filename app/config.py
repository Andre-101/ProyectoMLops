import os
from pathlib import Path

class Settings:
    # ID del modelo (puede ser repo de HF o ruta local)
    MODEL_ID: str = os.getenv("MODEL_ID", "")

    # Tarea del pipeline de Hugging Face (no la usamos para if/else, solo se pasa al pipeline)
    # Ej: "ner", "zero-shot-classification", "text-classification", etc.
    MODEL_TASK: str = os.getenv("MODEL_TASK", "text-classification")

    # Clase concreta del modelo ONNX (totalmente configurable)
    # Ej NER: "optimum.onnxruntime.ORTModelForTokenClassification"
    # Ej ZS : "optimum.onnxruntime.ORTModelForSequenceClassification"
    MODEL_CLASS: str = os.getenv("MODEL_CLASS", "")

    # Tokenizer (si lo quieres separar del modelo)
    TOKENIZER_ID: str = os.getenv("TOKENIZER_ID", "")

    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "dev")
    LOG_DIR: Path = Path(os.getenv("LOG_DIR", "/logs"))

settings = Settings()
