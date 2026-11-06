"""Converte o dataset Hard Hat Detection (formato FiftyOne) para o formato YOLO, com as nossas classes.

Origem: https://huggingface.co/datasets/Voxel51/hard-hat-detection (CC0)
  - samples.json: anotações de todas as imagens
  - data/*.png:   imagens (baixamos só as que têm a classe "head")

Exemplo:
    python src/scripts/convert_hardhat.py
"""

import argparse
import json
import shutil
from pathlib import Path

# nome no dataset de origem -> índice no nosso data.yaml
# (0 helmet, 1 no-helmet, 2 no-vest, 3 person, 4 vest)
CLASS_MAP = {
    "helmet": 0,
    "head": 1,  # cabeça sem capacete = no-helmet
    # "person" fica de fora de propósito: só ~750 caixas em 5000 imagens (anotação incompleta).
    # As pessoas serão marcadas pelo pseudo-rótulo (build_merged_dataset.py).
}


def to_yolo_line(class_id: int, bbox: list[float]) -> str:
    """FiftyOne [x_esquerda, y_topo, largura, altura] -> YOLO "classe x_centro y_centro largura altura".

    Os dois formatos usam coordenadas normalizadas (0 a 1); só muda o ponto de referência:
    canto superior esquerdo (FiftyOne) vs. centro da caixa (YOLO).
    """
    x, y, w, h = bbox
    x_center = x + w / 2
    y_center = y + h / 2
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


def convert(src: Path, dst: Path) -> tuple[int, dict[str, int]]:
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)

    samples = json.loads((src / "samples.json").read_text())["samples"]

    converted = 0
    box_counts = {name: 0 for name in CLASS_MAP}
    for sample in samples:
        image_path = src / sample["filepath"]  # ex: datasets/hardhat_raw/data/hard_hat_workers0.png
        if not image_path.exists():            # só baixamos as imagens com "head"
            continue

        detections = (sample.get("ground_truth") or {}).get("detections", [])
        lines = []
        for det in detections:
            if det["label"] in CLASS_MAP:
                lines.append(to_yolo_line(CLASS_MAP[det["label"]], det["bounding_box"]))
                box_counts[det["label"]] += 1

        shutil.copy(image_path, dst / "images" / image_path.name)
        # O .txt precisa ter o mesmo nome da imagem: é assim que o YOLO associa os dois.
        (dst / "labels" / f"{image_path.stem}.txt").write_text("\n".join(lines) + "\n")
        converted += 1

    return converted, box_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converte o Hard Hat Detection (FiftyOne) para YOLO")
    parser.add_argument("--src", default="datasets/hardhat_raw", help="Pasta com samples.json e data/")
    parser.add_argument("--dst", default="datasets/hardhat_yolo", help="Pasta de saída (images/ e labels/)")
    args = parser.parse_args()

    n, counts = convert(Path(args.src), Path(args.dst))
    print(f"{n} imagens convertidas em {args.dst}")
    print("Caixas por classe:", counts)
