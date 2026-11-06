"""Mede precisão e recall de uma classe em vários limiares de confiança, para escolher o limiar do infer.py.

O mAP resume todos os limiares; na prática, o detector usa UM limiar (ex.: 0.5) e o que importa é:
    precisão = dos alertas disparados, quantos estavam certos   (alarmes falsos)
    recall   = dos casos reais, quantos geraram alerta            (casos perdidos)

Uma detecção conta como acerto quando tem IoU >= 0.5 com uma caixa real da mesma classe
(cada caixa real só pode ser "encontrada" uma vez).

Exemplo:
    python src/scripts/tune_threshold.py --weights weights/best_v3.pt --class-name no-helmet
"""

import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]


def read_ground_truth(label_file: Path, class_id: int) -> np.ndarray:
    """Caixas reais da classe, em xyxy normalizado. Aceita linhas de caixa (5 valores) e de polígono."""
    boxes = []
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            values = line.split()
            if not values or int(values[0]) != class_id:
                continue
            coords = list(map(float, values[1:]))
            if len(coords) == 4:  # caixa: x_centro y_centro largura altura
                xc, yc, w, h = coords
                boxes.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
            else:                 # polígono: x1 y1 x2 y2 ... -> retângulo que o envolve
                xs, ys = coords[0::2], coords[1::2]
                boxes.append([min(xs), min(ys), max(xs), max(ys)])
    return np.array(boxes).reshape(-1, 4)


def iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0]); y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2]); y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(box) + area(others) - inter + 1e-9)


def count_matches(preds: np.ndarray, scores: np.ndarray, gt: np.ndarray, threshold: float) -> tuple[int, int, int]:
    """Devolve (acertos, alarmes falsos, casos perdidos) num limiar."""
    keep = scores >= threshold
    preds, scores = preds[keep], scores[keep]
    matched = np.zeros(len(gt), dtype=bool)
    tp = 0
    for p in preds[np.argsort(-scores)]:  # as mais confiantes escolhem primeiro
        if len(gt):
            ious = iou(p, gt)
            ious[matched] = 0
            best = ious.argmax()
            if ious[best] >= 0.5:
                matched[best] = True
                tp += 1
    return tp, len(preds) - tp, len(gt) - tp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precisão/recall de uma classe por limiar de confiança")
    parser.add_argument("--weights", default="weights/best_v3.pt")
    parser.add_argument("--data-dir", default="datasets/construction-safety/test",
                        help="Split com images/ e labels/")
    parser.add_argument("--class-name", default="no-helmet")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO(args.weights)
    class_id = {name: i for i, name in model.names.items()}[args.class_name]
    data_dir = Path(args.data_dir)

    # Uma inferência por imagem com limiar baixo; os limiares maiores são só filtros sobre ela.
    per_image = []
    for image in sorted((data_dir / "images").iterdir()):
        r = model(image, conf=min(THRESHOLDS), classes=[class_id], device=args.device, verbose=False)[0]
        gt = read_ground_truth(data_dir / "labels" / f"{image.stem}.txt", class_id)
        per_image.append((r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy(), gt))

    total_gt = sum(len(gt) for _, _, gt in per_image)
    print(f"Modelo: {args.weights} | classe: {args.class_name} | {total_gt} caixas reais em {len(per_image)} imagens\n")
    print("| Limiar | Precisão | Recall | Acertos | Alarmes falsos | Perdidos |")
    print("|---|---|---|---|---|---|")
    for t in THRESHOLDS:
        tp, fp, fn = map(sum, zip(*(count_matches(p, s, gt, t) for p, s, gt in per_image)))
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        print(f"| {t:.2f} | {precision:.2f} | {recall:.2f} | {tp} | {fp} | {fn} |")
