from functools import lru_cache
from importlib import import_module
from typing import Any, Dict

from transformers import AutoTokenizer, pipeline
from .config import settings


def _load_model_class():
    """
    Carga la clase de modelo a partir del string MODEL_CLASS.
    Ejemplo:
      MODEL_CLASS="optimum.onnxruntime.ORTModelForTokenClassification"
    """
    if not settings.MODEL_CLASS:
        return None  # usaremos el pipeline normal de HF (no ONNX)

    module_name, class_name = settings.MODEL_CLASS.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


@lru_cache(maxsize=1)
def get_model_pipeline():
    """
    Crea un único pipeline genérico en función de:
      - MODEL_ID
      - MODEL_TASK
      - MODEL_CLASS
      - TOKENIZER_ID

    No sabe si es NER, Zero-Shot, sentiment, etc.
    """
    model_id = settings.MODEL_ID
    if not model_id:
        raise ValueError("MODEL_ID no está configurado")

    task = settings.MODEL_TASK

    tokenizer_id = settings.TOKENIZER_ID or model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    ModelClass = _load_model_class()

    if ModelClass is None:
        # Modo "normal" (no ONNX), por si algún día quieres usar otro tipo de modelo
        clf = pipeline(
            task=task,
            model=model_id,
            tokenizer=tokenizer,
        )
    else:
        # Modo ONNX usando la clase que definas en MODEL_CLASS
        model = ModelClass.from_pretrained(model_id)
        clf = pipeline(
            task=task,
            model=model,
            tokenizer=tokenizer,
        )

    return clf


def run_inference(payload: Dict[str, Any]) -> Any:
    """
    Función genérica: recibe un dict y se lo pasa al pipeline.
    La API no sabe de labels, tokens ni nada de eso.

    Convención:
      - payload["inputs"]  -> texto o lista de textos
      - payload["parameters"] -> dict con parámetros extra (candidate_labels, etc.)
    """
    pipe = get_model_pipeline()

    inputs = payload.get("inputs")
    parameters = payload.get("parameters") or {}

    # Ejemplos:
    #  NER: pipe("My name is John", aggregation_strategy="simple")
    #  Zero-shot: pipe("I love transformers", candidate_labels=[...])
    result = pipe(inputs, **parameters)
    return result
