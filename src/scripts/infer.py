import argparse
import sys
from ppe_detector.detector import PPEDetector
from ppe_detector.video import process_video


def parse_class_conf(items: list[str]) -> dict[str, float]:
    """["no-helmet=0.3", "no-vest=0.4"] -> {"no-helmet": 0.3, "no-vest": 0.4}"""
    class_conf = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Formato inválido em --class-conf: '{item}' (use classe=valor, ex.: no-helmet=0.3)")
        class_conf[name.strip()] = float(value)
    return class_conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detector de EPIs em Vídeo usando YOLOv8")

    parser.add_argument("--source", type=str, required=True, help="Caminho do vídeo de entrada (ex: video.mp4)")
    parser.add_argument("--model", type=str, default="weights/best_v3.pt", help="Caminho do arquivo de pesos .pt ou .onnx")
    parser.add_argument("--output", type=str, default="output_video.mp4", help="Nome do arquivo de saída")
    parser.add_argument("--conf", type=float, default=0.5, help="Limiar de confiança (0.0 a 1.0)")
    # Limiar mais baixo para "sem capacete": no teste, o recall dessa classe sobe de 0.46 para 0.67
    # (precisão de 0.85 para 0.67). Escolhido com src/scripts/tune_threshold.py. Use --class-conf sem
    # valores para desligar e usar o --conf em todas as classes.
    parser.add_argument("--class-conf", type=str, nargs="*", default=["no-helmet=0.3"],
                        help="Limiares por classe, ex.: --class-conf no-helmet=0.3 no-vest=0.4")
    parser.add_argument("--device", type=str, default="cpu", help="cpu ou 0 para gpu")
    parser.add_argument("--show", action="store_true", help="Mostra o vídeo numa janela")

    args = parser.parse_args()
    try:
        # Criando o detector
        detector = PPEDetector(model_path=args.model, device=args.device, conf=args.conf,
                               class_conf=parse_class_conf(args.class_conf))
        frame_processed = process_video(detector=detector, source=args.source, output_path=args.output, show=args.show)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Erro: {e}")
        sys.exit(1)
