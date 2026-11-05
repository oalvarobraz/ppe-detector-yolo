"""Exporta o modelo treinado (.pt) para ONNX ou OpenVINO.

Exemplos:
    python src/scripts/export_model.py                     # ONNX  -> weights/best.onnx
    python src/scripts/export_model.py --format openvino   # OpenVINO -> weights/best_openvino_model/
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporta o modelo .pt para ONNX ou OpenVINO")
    parser.add_argument("--model", type=str, default="weights/best.pt", help="Caminho do arquivo de pesos .pt")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "openvino"], help="Formato de saída")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Erro: arquivo de pesos não encontrado: {args.model}")
        sys.exit(1)

    model = YOLO(args.model)
    if args.format == "onnx":
        # simplify: remove operações redundantes | dynamic=False: entrada fixa 640x640 (mais rápido na CPU)
        exported = model.export(format="onnx", imgsz=args.imgsz, simplify=True, dynamic=False)
    else:
        # half=False / int8=False: mantém FP32, para comparar de igual para igual com o .pt e o .onnx
        exported = model.export(format="openvino", imgsz=args.imgsz, half=False, int8=False, dynamic=False)

    print(f"Modelo exportado para: {exported}")
