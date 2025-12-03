# tests/test_inference_basic.py
import json
import os
from pathlib import Path
from typing import Any, Dict

from app.config import settings
from app.inference import get_model_pipeline, run_inference


def load_test_data() -> list[Dict[str, Any]]:
    """
    Carga el dataset de prueba desde TEST_DATA_PATH.
    Formato esperado: lista de casos, cada uno un dict con al menos:
      - "inputs": texto o lista de textos
      - "parameters": dict opcional con parámetros extra (candidate_labels, etc.)
      - campos opcionales para aserciones suaves, por ejemplo:
          - "min_entities" (NER)
          - "expected_label" (zero-shot u otros)
    """
    path_str = os.getenv("TEST_DATA_PATH", "tests_data/test_data.json")
    path = Path(path_str)
    assert path.is_file(), f"TEST_DATA_PATH no existe: {path}"

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list) and len(data) > 0, "test_data.json debe ser una lista no vacía"
    return data


def test_model_loads():
    """
    Prueba mínima: el pipeline se construye sin explotar.
    Esto valida que MODEL_ID, MODEL_TASK, MODEL_CLASS y TOKENIZER_ID
    son coherentes.
    """
    pipe = get_model_pipeline()
    assert pipe is not None


def test_single_example_behaves_reasonably():
    """
    Toma el primer ejemplo del dataset de prueba y verifica que:
      - el modelo devuelve algo no vacío
      - si el dataset define pistas (min_entities, expected_label), 
        se usan para una aserción suave, sin acoplarse a un modelo específico.
    """
    data = load_test_data()
    example = data[0]

    payload = {
        "inputs": example["inputs"],
        "parameters": example.get("parameters") or {},
    }

    result = run_inference(payload)

    # Aserción genérica: no debe ser None ni estar vacío
    assert result is not None, "El resultado de inferencia no debe ser None"
    if isinstance(result, (list, tuple)):
        assert len(result) > 0, "La salida de una lista de resultados no debe estar vacía"

    # Aserciones suaves según el tipo de tarea / pistas del JSON
    task = settings.MODEL_TASK

    # Caso típico NER: salida = lista de entidades
    min_entities = example.get("min_entities")
    if task == "ner" and min_entities is not None:
        assert isinstance(result, list), "Para NER se espera una lista de entidades"
        assert len(result) >= min_entities, (
            f"Se esperaban al menos {min_entities} entidades, se obtuvo {len(result)}"
        )

    # Caso típico zero-shot: salida = dict con labels/scores
    expected_label = example.get("expected_label")
    if task == "zero-shot-classification" and expected_label is not None:
        assert isinstance(result, dict), "Para zero-shot se espera un dict"
        labels = result.get("labels") or []
        assert expected_label in labels, (
            f"Se esperaba que '{expected_label}' esté entre las labels {labels}"
        )
