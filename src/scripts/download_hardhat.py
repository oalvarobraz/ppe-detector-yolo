"""Baixa do Hugging Face as imagens do Hard Hat Detection que contêm a classe "head" (sem capacete).

Dataset: https://huggingface.co/datasets/Voxel51/hard-hat-detection (CC0, 5000 imagens)
Só ~920 imagens têm "head"; baixamos apenas essas (~225 MB) em vez das 5000.

Exemplo:
    python src/scripts/download_hardhat.py
"""

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE_URL = "https://huggingface.co/datasets/Voxel51/hard-hat-detection/resolve/main"


def download(url: str, dst: Path) -> None:
    if dst.exists():  # permite retomar um download interrompido
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dst)  # só aparece com o nome final quando terminou de baixar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baixa as imagens com 'head' do Hard Hat Detection")
    parser.add_argument("--dst", default="datasets/hardhat_raw")
    parser.add_argument("--workers", type=int, default=8, help="Downloads em paralelo")
    args = parser.parse_args()

    dst = Path(args.dst)
    download(f"{BASE_URL}/samples.json", dst / "samples.json")
    samples = json.loads((dst / "samples.json").read_text())["samples"]

    files = [
        s["filepath"] for s in samples
        if any(d["label"] == "head" for d in (s.get("ground_truth") or {}).get("detections", []))
    ]
    print(f"{len(files)} imagens com 'head' de {len(samples)} no total")

    with ThreadPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(lambda f: download(f"{BASE_URL}/{f}", dst / f), files), 1):
            if i % 100 == 0:
                print(f"Baixadas: {i}/{len(files)}", end="\r")
    print(f"\nPronto: {dst}")
