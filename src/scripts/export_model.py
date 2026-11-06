"""Exporta o modelo treinado (.pt) para ONNX ou OpenVINO, em FP32 ou INT8.

Exemplos:
    python src/scripts/export_model.py                     # ONNX  -> weights/best.onnx
    python src/scripts/export_model.py --format openvino   # OpenVINO -> weights/best_openvino_model/

    # INT8 (quantizado): precisa de imagens de calibração
    python src/scripts/export_model.py --model weights/best_v3.pt --format openvino --int8 \
        --data datasets/merged/data.yaml                   # -> weights/best_v3_int8_openvino_model/
    python src/scripts/export_model.py --model weights/best_v3.pt --format onnx --int8 \
        --data datasets/merged/data.yaml                   # -> weights/best_v3_int8.onnx

Sobre o INT8:
    Os pesos e as ativações passam de números de 32 bits (float) para inteiros de 8 bits. O modelo fica
    ~4x menor e, em CPU, costuma ficar mais rápido. Para escolher a escala de cada camada, o quantizador
    roda a rede em algumas imagens ("calibração") e mede a faixa de valores. Usamos imagens do TREINO,
    nunca do teste, para a avaliação continuar limpa.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporta o modelo .pt para ONNX ou OpenVINO (FP32 ou INT8)")
    parser.add_argument("--model", type=str, default="weights/best.pt", help="Caminho do arquivo de pesos .pt")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "openvino"], help="Formato de saída")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--int8", action="store_true", help="Quantiza para INT8 (precisa de --data)")
    parser.add_argument("--data", type=str, default=None, help="data.yaml com as imagens de calibração do INT8")
    parser.add_argument("--split", type=str, default="train", help="Split usado na calibração (nunca o de teste)")
    # 0.16 do treino do dataset ampliado = ~300 imagens (o mínimo recomendado pelo ultralytics).
    # Com 0.25, a exportação ONNX INT8 foi morta por falta de memória num notebook de 11 GB.
    parser.add_argument("--fraction", type=float, default=0.16,
                        help="Fração do split usada na calibração (0.16 do dataset ampliado = ~300 imagens)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Erro: arquivo de pesos não encontrado: {model_path}")
        sys.exit(1)
    if args.int8 and not args.data:
        print("Erro: --int8 precisa de --data (as imagens de calibração)")
        sys.exit(1)

    # dynamic=False: entrada fixa 640x640 (mais rápido na CPU) | quantize=32 (FP32) ou 8 (INT8)
    common = {"imgsz": args.imgsz, "dynamic": False, "quantize": 8 if args.int8 else 32}
    if args.int8:
        common.update(data=args.data, split=args.split, fraction=args.fraction)

    if args.format == "onnx" and args.int8:
        # O ultralytics gera o .onnx FP32, cria o _int8.onnx a partir dele e APAGA o FP32. Para não destruir
        # um best.onnx FP32 que já exista, a exportação acontece numa cópia do .pt em pasta temporária.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_model = Path(tmp) / model_path.name
            shutil.copy(model_path, tmp_model)
            produced = Path(YOLO(tmp_model).export(format="onnx", simplify=True, **common))
            exported = model_path.with_name(produced.name)  # ex.: weights/best_v3_int8.onnx
            shutil.move(produced, exported)
    elif args.format == "onnx":
        # simplify: remove operações redundantes do grafo
        exported = YOLO(model_path).export(format="onnx", simplify=True, **common)
    else:
        exported = YOLO(model_path).export(format="openvino", **common)

    print(f"Modelo exportado para: {exported}")
