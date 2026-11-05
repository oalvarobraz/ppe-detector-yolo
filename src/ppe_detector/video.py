"""Loop de vídeo: lê frames da fonte, detecta, desenha o painel e grava/mostra."""

from pathlib import Path

import cv2

from .detector import PPEDetector
from .overlay import draw_counts_panel

WINDOW_NAME = "PPE Detector"


def open_source(source: str) -> cv2.VideoCapture:
    """Abre um arquivo de vídeo ou uma webcam ("0", "1", ...)."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        if not Path(source).exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {source}")
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {source}")
    return cap


def process_video(
    detector: PPEDetector,
    source: str,
    output_path: str | None = None,
    show: bool = False,
) -> int:
    """Processa a fonte inteira e devolve quantos frames foram processados.

    - output_path: se informado, grava o vídeo anotado.
    - show: abre uma janela com o resultado (tecla 'q' encerra).
    Ctrl+C também encerra, e o vídeo gravado até ali é salvo corretamente.
    """
    cap = open_source(source)
    is_webcam = source.isdigit()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # algumas webcams informam 0
    total_frames = -1 if is_webcam else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            r = detector.predict(frame)
            annotated = draw_counts_panel(r.result.plot(), r.counts)
            frame_count += 1

            if writer:
                writer.write(annotated)

            if show:
                cv2.imshow(WINDOW_NAME, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_count % 30 == 0:
                _print_progress(frame_count, total_frames)

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário (Ctrl+C).")

    finally:
        # Roda SEMPRE (fim do vídeo, 'q', Ctrl+C ou erro): sem o release() o mp4 fica corrompido.
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    print(f"\nFrames processados: {frame_count}")
    return frame_count


def _print_progress(frame_count: int, total_frames: int) -> None:
    if total_frames > 0:
        print(f"Processado: {frame_count}/{total_frames} ({frame_count / total_frames:.1%})", end="\r")
    else:  # webcam: não tem total
        print(f"Processado: {frame_count} frames", end="\r")
