"""Monta um conjunto de teste extra, focado em "sem capacete", com imagens que NENHUM modelo viu no treino.

Por quê: o teste original tem só 24 caixas de no-helmet, poucas demais para métricas confiáveis.

Fonte: Hard Hat Workers v5 no Roboflow (joseph-nelson/hard-hat-workers, domínio público, 7035 imagens,
classes head/helmet). É a coleção da qual o dataset do Hugging Face usado no treino (5000 imagens) é um
recorte. Por isso a maioria das imagens com "head" já está no treino, e é preciso tirar as repetidas.

Como as repetidas são encontradas:
    Hash perceptual (dHash): a imagem vira 9x8 pixels cinza e cada bit diz se um pixel é mais claro que o
    vizinho; imagens iguais dão hashes quase iguais. As imagens do Hugging Face foram redimensionadas para
    416x416 com bordas espelhadas, então cada imagem do Roboflow é comparada de dois jeitos: como está e
    com esse mesmo redimensionamento. Distância <= 8 bits (de 64) = mesma imagem (limite escolhido olhando
    a distribuição: há uma lacuna clara entre as repetidas, 0-7, e as cenas diferentes, 9+).

O que entra:
    - todas as imagens inéditas com "head"  -> medem o recall do alerta
    - N imagens inéditas só com capacete     -> medem alarmes falsos em quem está de capacete
    Só helmet e no-helmet são anotados nessas imagens: as métricas de person/vest/no-vest não valem aqui.

Exemplo:
    python src/scripts/build_test_nohelmet.py
"""

import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# Hard Hat Workers v5: 0 head, 1 helmet  ->  nosso data.yaml: 0 helmet, 1 no-helmet
CLASS_REMAP = {0: 1, 1: 0}
OUR_CLASS_NAMES = ["helmet", "no-helmet", "no-vest", "person", "vest"]
HEAD = 0
DUPLICATE_MAX_DISTANCE = 8
SPLITS = ("train", "valid", "test")


def dhash(img: np.ndarray) -> np.ndarray:
    """Hash perceptual de 64 bits (como vetor de 0/1)."""
    gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    return (gray[:, 1:] > gray[:, :-1]).flatten().astype(np.uint8)


def letterbox_reflect(img: np.ndarray, size: int = 416) -> np.ndarray:
    """Reproduz o pré-processamento das imagens do Hugging Face: cabe em 416x416, bordas espelhadas."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = round(w * scale), round(h * scale)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top, left = (size - nh) // 2, (size - nw) // 2
    return cv2.copyMakeBorder(img, top, size - nh - top, left, size - nw - left, cv2.BORDER_REFLECT)


def min_distance(hashes: np.ndarray, reference: np.ndarray) -> int:
    """Menor distância (bits diferentes) entre qualquer um dos hashes e o conjunto de referência."""
    if len(reference) == 0:
        return 64
    return int(min((reference != h).sum(axis=1).min() for h in hashes))


def download_if_missing(dst: Path) -> None:
    if dst.exists():
        return
    from dotenv import load_dotenv
    from roboflow import Roboflow

    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Erro: ROBOFLOW_API_KEY não encontrada no .env")
    project = Roboflow(api_key=api_key).workspace("joseph-nelson").project("hard-hat-workers")
    project.version(5).download("yolov8", location=str(dst))


def read_labels(label_file: Path) -> list[list[str]]:
    if not label_file.exists():
        return []
    return [line.split() for line in label_file.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monta o teste extra de no-helmet sem vazamento do treino")
    parser.add_argument("--src", default="datasets/hardhat_workers_rf", help="Hard Hat Workers v5 (baixa se faltar)")
    parser.add_argument("--seen", nargs="+",
                        default=["datasets/hardhat_yolo/images", "datasets/construction-safety/train/images",
                                 "datasets/construction-safety/valid/images", "datasets/construction-safety/test/images"],
                        help="Pastas de imagens já usadas (treino e avaliações existentes)")
    parser.add_argument("--dst", default="datasets/test_nohelmet")
    parser.add_argument("--negatives", type=int, default=1000, help="Quantas imagens inéditas só com capacete")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    download_if_missing(src)

    print("Calculando hashes das imagens já usadas...")
    seen = np.array([dhash(cv2.imread(str(p))) for d in args.seen for p in sorted(Path(d).iterdir())])
    print(f"  {len(seen)} imagens")

    print("Procurando imagens inéditas no Hard Hat Workers...")
    with_head, helmet_only = [], []
    kept_hashes: list[np.ndarray] = []  # para também descartar repetidas dentro do próprio conjunto novo
    n_duplicates = 0
    images = [p for s in SPLITS for p in sorted((src / s / "images").iterdir())]
    for i, image_path in enumerate(images, 1):
        img = cv2.imread(str(image_path))
        hashes = np.array([dhash(img), dhash(letterbox_reflect(img))])
        if min(min_distance(hashes, seen), min_distance(hashes, np.array(kept_hashes))) <= DUPLICATE_MAX_DISTANCE:
            n_duplicates += 1
            continue
        kept_hashes.append(hashes[0])
        labels = read_labels(image_path.parent.parent / "labels" / f"{image_path.stem}.txt")
        (with_head if any(int(l[0]) == HEAD for l in labels) else helmet_only).append((image_path, labels))
        if i % 500 == 0:
            print(f"  {i}/{len(images)}", end="\r")
    print(f"\n  repetidas descartadas: {n_duplicates} | inéditas com head: {len(with_head)} "
          f"| inéditas só com capacete: {len(helmet_only)}")

    # Amostra das imagens só com capacete: ordenadas por nome antes do sorteio, para ser reprodutível.
    helmet_only.sort(key=lambda x: x[0].name)
    negatives = random.Random(args.seed).sample(helmet_only, min(args.negatives, len(helmet_only)))
    selected = with_head + negatives

    if dst.exists():
        shutil.rmtree(dst)
    (dst / "images").mkdir(parents=True)
    (dst / "labels").mkdir(parents=True)
    counts = {name: 0 for name in OUR_CLASS_NAMES}
    for image_path, labels in selected:
        shutil.copy(image_path, dst / "images" / image_path.name)
        lines = []
        for cls, *coords in labels:
            new_cls = CLASS_REMAP[int(cls)]
            counts[OUR_CLASS_NAMES[new_cls]] += 1
            lines.append(" ".join([str(new_cls), *coords]))
        (dst / "labels" / f"{image_path.stem}.txt").write_text("\n".join(lines) + "\n")

    # O ultralytics exige train/val no yaml; aqui tudo aponta para o mesmo conjunto (só avaliação).
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(OUR_CLASS_NAMES))
    (dst / "data.yaml").write_text(f"train: images\nval: images\ntest: images\nnames:\n{names}\n")
    # Lista das imagens escolhidas, para conferir que o conjunto é o mesmo em outra máquina.
    (dst / "manifest.txt").write_text("\n".join(sorted(p.name for p, _ in selected)) + "\n")

    print(f"\nTeste extra em {dst}: {len(selected)} imagens "
          f"({len(with_head)} com head + {len(negatives)} só com capacete)")
    print(f"  caixas: helmet={counts['helmet']}  no-helmet={counts['no-helmet']}")
