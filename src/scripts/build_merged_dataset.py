"""Junta o dataset original com o extra (Hard Hat), completando os rótulos que faltam com pseudo-rótulos.

Por que pseudo-rótulo?
    O dataset extra só anota capacete (helmet) e cabeça sem capacete (no-helmet). Se as imagens entrassem
    assim, todo colete e toda pessoa nelas ficariam SEM caixa, e o modelo aprenderia que "colete é fundo".
    Então rodamos um modelo já treinado (best_v2.pt) nessas imagens e aproveitamos as detecções de
    person / vest / no-vest com confiança alta. helmet / no-helmet continuam vindo da anotação real.

Resultado (datasets/merged):
    train/ = treino original + imagens extras (prefixo "hh_")
    valid/ = validação original, intacta
    test/  = teste original, intacto   -> a comparação com os modelos anteriores continua justa

Exemplo:
    python src/scripts/build_merged_dataset.py --device 0
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

CLASS_NAMES = ["helmet", "no-helmet", "no-vest", "person", "vest"]
REAL_CLASSES = {0, 1}         # helmet, no-helmet: vêm da anotação do dataset extra
PSEUDO_CLASSES = [2, 3, 4]    # no-vest, person, vest: vêm do modelo
EXTRA_PREFIX = "hh_"          # evita colisão de nomes com as imagens originais
SPLITS = ["train", "valid", "test"]


def copy_split(src_split: Path, dst_split: Path) -> int:
    """Copia images/ e labels/ de um split do dataset original."""
    for sub in ("images", "labels"):
        shutil.copytree(src_split / sub, dst_split / sub, dirs_exist_ok=True)
    return len(list((dst_split / "images").iterdir()))


def pseudo_label_lines(model: YOLO, image: Path, conf: float, device: str) -> list[str]:
    """Roda o modelo na imagem e devolve linhas YOLO só para as classes de pseudo-rótulo."""
    result = model(image, conf=conf, classes=PSEUDO_CLASSES, device=device, verbose=False)[0]
    lines = []
    for cls_id, (xc, yc, w, h) in zip(result.boxes.cls.int().tolist(), result.boxes.xywhn.tolist()):
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def add_extra(extra: Path, dst_train: Path, model: YOLO, conf: float, device: str) -> Counter:
    counts = Counter()
    images = sorted((extra / "images").iterdir())
    for i, image in enumerate(images, 1):
        real_lines = (extra / "labels" / f"{image.stem}.txt").read_text().split()
        # O .txt real tem 5 valores por linha; reagrupa e mantém só helmet / no-helmet.
        real = [" ".join(real_lines[j:j + 5]) for j in range(0, len(real_lines), 5)]
        real = [line for line in real if int(line.split()[0]) in REAL_CLASSES]

        pseudo = pseudo_label_lines(model, image, conf, device)

        name = f"{EXTRA_PREFIX}{image.stem}"
        shutil.copy(image, dst_train / "images" / f"{name}{image.suffix}")
        (dst_train / "labels" / f"{name}.txt").write_text("\n".join(real + pseudo) + "\n")

        for line in real + pseudo:
            counts[CLASS_NAMES[int(line.split()[0])]] += 1
        if i % 100 == 0:
            print(f"Pseudo-rótulo: {i}/{len(images)}", end="\r")
    print()
    return counts


def count_labels(labels_dir: Path) -> Counter:
    counts = Counter()
    for txt in labels_dir.glob("*.txt"):
        for line in txt.read_text().splitlines():
            if line.strip():
                counts[CLASS_NAMES[int(line.split()[0])]] += 1
    return counts


def write_yaml(dst: Path) -> None:
    # Sem "path:", o ultralytics resolve train/val/test a partir da pasta do próprio data.yaml.
    # Assim a pasta pode ser copiada para o Colab (ou qualquer lugar) sem editar nada.
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    (dst / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\n"
        f"names:\n{names}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Junta o dataset original com o Hard Hat + pseudo-rótulos")
    parser.add_argument("--base", default="datasets/construction-safety", help="Dataset original (Roboflow)")
    parser.add_argument("--extra", default="datasets/hardhat_yolo", help="Saída do convert_hardhat.py")
    parser.add_argument("--dst", default="datasets/merged")
    parser.add_argument("--weights", default="weights/best_v2.pt", help="Modelo usado no pseudo-rótulo")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confiança mínima do pseudo-rótulo (maior = menos caixas erradas, mais faltando)")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    base, extra, dst = Path(args.base), Path(args.extra), Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)  # sempre recria do zero, para não misturar com uma execução anterior

    for split in SPLITS:
        n = copy_split(base / split, dst / split)
        print(f"{split}: {n} imagens originais copiadas")

    model = YOLO(args.weights)
    pseudo_counts = add_extra(extra, dst / "train", model, args.conf, args.device)
    print(f"Caixas adicionadas pelo dataset extra (conf >= {args.conf}): {dict(pseudo_counts)}")

    write_yaml(dst)

    print("\nResumo do dataset final:")
    for split in SPLITS:
        n_img = len(list((dst / split / "images").iterdir()))
        counts = count_labels(dst / split / "labels")
        print(f"  {split:5s}: {n_img:5d} imagens | " + "  ".join(f"{c}={counts[c]}" for c in CLASS_NAMES))
    print(f"\ndata.yaml: {dst / 'data.yaml'}")
