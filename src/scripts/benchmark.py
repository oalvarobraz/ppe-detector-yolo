"""Compara velocidade (e, opcionalmente, mAP) entre variantes do modelo.

Variantes: PyTorch CPU, PyTorch GPU e ONNX Runtime CPU.

Exemplo:
    PYTHONPATH=src .venv/bin/python src/scripts/benchmark.py --video videos/EPI.mp4
    PYTHONPATH=src .venv/bin/python src/scripts/benchmark.py --video videos/EPI.mp4 \
        --data datasets/construction-safety/data.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ppe_detector.detector import PPEDetector

def build_variants(pt_path: str) -> list[tuple[str, str, str]]:
    """(nome que aparece na tabela, arquivo de pesos, device) para cada variante.

    Os exportados ficam ao lado do .pt: best.pt -> best.onnx e best_openvino_model/.
    """
    pt = Path(pt_path)
    return [
        ("PyTorch CPU", str(pt), "cpu"),
        ("PyTorch GPU", str(pt), "0"),
        ("ONNX Runtime CPU", str(pt.with_suffix(".onnx")), "cpu"),
        # "intel:cpu" e não "cpu": com "cpu" o ultralytics deixa o OpenVINO em modo AUTO,
        # que pode escolher a GPU NVIDIA via OpenCL em vez da CPU.
        ("OpenVINO CPU", str(pt.parent / f"{pt.stem}_openvino_model"), "intel:cpu"),
    ]


def load_frames(video_path: str, max_frames: int) -> list[np.ndarray]:
    """Carrega os frames na memória ANTES de medir, para a leitura do vídeo não entrar no tempo."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def warmup(detector: PPEDetector, frames: list[np.ndarray], n: int) -> None:
    # As primeiras inferências são lentas (alocação de memória, compilação de kernels).
    for frame in frames[:n]:
        detector.predict(frame)


def time_frames(detector: PPEDetector, frames: list[np.ndarray]) -> tuple[list[float], list[float]]:
    """Mede cada frame. Devolve (tempos só da inferência, tempos totais), em ms."""
    inference_ms, total_ms = [], []
    for frame in frames:
        start = time.perf_counter()
        r = detector.predict(frame)
        total_ms.append((time.perf_counter() - start) * 1000)  # tudo: pré + inferência + pós
        inference_ms.append(r.speed["inference"])              # só a rede neural
    return inference_ms, total_ms


def summarize(inference_ms: list[float], total_ms: list[float]) -> dict[str, float]:
    inference_ms, total_ms = np.array(inference_ms), np.array(total_ms)
    return {
        "inf_mean": inference_ms.mean(),
        "inf_p50": np.percentile(inference_ms, 50),
        "inf_p95": np.percentile(inference_ms, 95),
        "total_mean": total_ms.mean(),
        "fps": 1000 / total_ms.mean(),
    }


def measure_map(detector: PPEDetector, data_yaml: str, imgsz: int, split: str) -> dict:
    # batch=1 e rect=False porque o .onnx foi exportado com shape fixo 640x640 (dynamic=False).
    # O .pt, por padrão, valida com rect=True (padding mínimo por imagem), o que muda o resultado
    # e tornaria a comparação injusta. Com rect=False, todas as variantes recebem a mesma entrada.
    metrics = detector.model.val(
        data=data_yaml, imgsz=imgsz, batch=1, split=split, rect=False,
        device=detector.device, plots=False, verbose=False,
    )
    names = detector.model.names
    per_class = {names[c]: metrics.box.ap50[i] for i, c in enumerate(metrics.box.ap_class_index)}
    return {"map50": metrics.box.map50, "map50_95": metrics.box.map, "per_class_map50": per_class}


def build_report(results: list[dict], args, n_frames: int) -> str:
    lines = [
        "## Benchmark",
        "",
        f"- Modelo: `{args.weights}`",
        f"- Vídeo: `{args.video}` ({n_frames} frames × {args.rounds} rodadas alternadas, "
        f"aquecimento de {args.warmup})",
        f"- CPU: {_cpu_name()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}",
        "",
        "| Variante | Inferência média (ms) | p50 | p95 | Total/frame (ms) | FPS |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        s = r["speed"]
        lines.append(
            f"| {r['name']} | {s['inf_mean']:.1f} | {s['inf_p50']:.1f} | {s['inf_p95']:.1f} "
            f"| {s['total_mean']:.1f} | {s['fps']:.1f} |"
        )

    with_map = [r for r in results if r.get("map")]
    if with_map:
        classes = list(with_map[0]["map"]["per_class_map50"])
        lines += [
            "",
            f"## Precisão (split `{args.split}`)",
            "",
            "| Variante | mAP50 | mAP50-95 | " + " | ".join(classes) + " |",
            "|---" * (3 + len(classes)) + "|",
        ]
        for r in with_map:
            m = r["map"]
            per_class = " | ".join(f"{m['per_class_map50'][c]:.3f}" for c in classes)
            lines.append(f"| {r['name']} | {m['map50']:.3f} | {m['map50_95']:.3f} | {per_class} |")

    return "\n".join(lines) + "\n"


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "?"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark de velocidade/mAP: PyTorch vs ONNX")
    parser.add_argument("--weights", type=str, default="weights/best.pt",
                        help="Pesos .pt; o .onnx e o _openvino_model são procurados ao lado dele")
    parser.add_argument("--video", type=str, required=True, help="Vídeo usado para medir velocidade")
    parser.add_argument("--frames", type=int, default=300, help="Máximo de frames medidos")
    parser.add_argument("--warmup", type=int, default=20, help="Frames de aquecimento (não medidos)")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Rodadas alternando as variantes (para o aquecimento da CPU afetar todas igual)")
    parser.add_argument("--data", type=str, default=None, help="data.yaml do dataset (se informado, mede o mAP)")
    parser.add_argument("--split", type=str, default="val", help="Split do dataset para o mAP (val ou test)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=str, default="results/benchmark.md", help="Onde salvar o relatório")
    args = parser.parse_args()

    try:
        frames = load_frames(args.video, args.frames)
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        sys.exit(1)
    print(f"Info: {len(frames)} frames carregados de {args.video}")

    # 1. Carrega e aquece todas as variantes
    detectors = {}
    for name, weights, device in build_variants(args.weights):
        if device.isdigit() and not torch.cuda.is_available():  # "0", "1"... = GPU CUDA
            print(f"Pulando {name}: GPU não disponível")
            continue
        try:
            detectors[name] = PPEDetector(weights, device=device)
        except FileNotFoundError as e:
            print(f"Pulando {name}: {e}")
            continue
        warmup(detectors[name], frames, args.warmup)

    # 2. Velocidade: rodadas alternadas (A, B, C, A, B, C, ...)
    timings = {name: ([], []) for name in detectors}
    for i in range(1, args.rounds + 1):
        for name, detector in detectors.items():
            inf, total = time_frames(detector, frames)
            timings[name][0].extend(inf)
            timings[name][1].extend(total)
            print(f"Rodada {i}/{args.rounds} | {name}: inferência mediana {np.median(inf):.1f} ms")

    results = [{"name": name, "speed": summarize(*timings[name])} for name in detectors]

    # 3. Precisão (opcional)
    if args.data:
        for result in results:
            print(f"\n=== mAP: {result['name']} ===")
            result["map"] = measure_map(detectors[result["name"]], args.data, args.imgsz, args.split)
            print(f"mAP50: {result['map']['map50']:.3f} | mAP50-95: {result['map']['map50_95']:.3f}")

    report = build_report(results, args, len(frames))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Relatório salvo em: {args.output}")
