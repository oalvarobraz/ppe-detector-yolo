"""Carrega o modelo YOLO (.pt ou .onnx) e conta as detecções de cada frame."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


@dataclass
class FrameResult:
    """O que o detector devolve para cada frame."""

    result: Results         # resultado bruto do ultralytics (caixas, .plot(), .speed)
    counts: dict[str, int]  # quantas detecções de cada classe, ex: {"helmet": 2, ...}

    @property
    def speed(self) -> dict[str, float]:
        """Tempo em ms de cada etapa: preprocess, inference, postprocess."""
        return self.result.speed


class PPEDetector:
    def __init__(self, model_path: str | Path, device: str = "cpu", conf: float = 0.5):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Arquivo de pesos não encontrado: {model_path}")

        # O YOLO() aceita .pt e .onnx com a mesma API. O task="detect" evita que o
        # ultralytics tente adivinhar a tarefa ao abrir um .onnx.
        self.model = YOLO(str(model_path), task="detect")
        self.device = device
        self.conf = conf

    @property
    def class_names(self) -> list[str]:
        return list(self.model.names.values())

    def predict(self, frame: np.ndarray) -> FrameResult:
        result = self.model(frame, conf=self.conf, device=self.device, verbose=False)[0]

        counts = dict.fromkeys(self.class_names, 0)
        for cls_id in result.boxes.cls.int().tolist():
            counts[self.model.names[cls_id]] += 1

        return FrameResult(result=result, counts=counts)
