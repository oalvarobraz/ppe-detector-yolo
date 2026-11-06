"""Carrega o modelo YOLO (.pt ou .onnx) e conta as detecções de cada frame."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
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
    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        conf: float = 0.5,
        class_conf: dict[str, float] | None = None,
    ):
        """
        conf:       limiar de confiança padrão, para todas as classes.
        class_conf: limiares específicos por classe, ex.: {"no-helmet": 0.3}.
                    Útil para dar mais sensibilidade aos alertas de segurança
                    (veja src/scripts/tune_threshold.py para escolher o valor).
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Arquivo de pesos não encontrado: {model_path}")

        # O YOLO() aceita .pt e .onnx com a mesma API. O task="detect" evita que o
        # ultralytics tente adivinhar a tarefa ao abrir um .onnx.
        self.model = YOLO(str(model_path), task="detect")
        self.device = device
        self.conf = conf
        self.class_conf = class_conf or {}

        unknown = set(self.class_conf) - set(self.class_names)
        if unknown:
            raise ValueError(f"Classes desconhecidas em class_conf: {sorted(unknown)}. "
                             f"Classes do modelo: {self.class_names}")

        # Limiar de cada índice de classe, para filtrar as caixas de uma vez só no predict().
        self._thresholds = torch.tensor(
            [self.class_conf.get(name, conf) for name in self.class_names], dtype=torch.float32
        )

    @property
    def class_names(self) -> list[str]:
        return list(self.model.names.values())

    def predict(self, frame: np.ndarray) -> FrameResult:
        # O modelo roda com o MENOR limiar; depois cada caixa é filtrada pelo limiar da sua classe.
        min_conf = float(self._thresholds.min())
        result = self.model(frame, conf=min_conf, device=self.device, verbose=False)[0]
        if self.class_conf:
            boxes = result.boxes
            keep = boxes.conf.cpu() >= self._thresholds[boxes.cls.cpu().long()]
            result = result[keep]

        counts = dict.fromkeys(self.class_names, 0)
        for cls_id in result.boxes.cls.int().tolist():
            counts[self.model.names[cls_id]] += 1

        return FrameResult(result=result, counts=counts)
