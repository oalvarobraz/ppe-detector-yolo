import cv2
import argparse
import sys
import os
from ultralytics import YOLO

def run_inference(source_path, model_path, output_path, conf_threshold, device_target):
    use_webcam = False

    if source_path.isdigit():
        use_webcam = True
        source = int(source_path)
        print(f"Info: Modo Webcam detectado: Câmera {source}")
    else:
        source = source_path
        if not os.path.exists(source):
            print(f"Erro: O arquivo de vídeo '{source}' não foi encontrado.")
            sys.exit(1)
        
    if not os.path.exists(model_path):
        print(f"Erro: O arquivo de pesos '{model_path}' não foi encontrado.")
        sys.exit(1)

    # Carrega o Modelo
    print(f"Info: Carregando modelo: {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Erro: Falha ao carregar o modelo YOLO: {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir a fonte de vídeo: {source}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if fps == 0: fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not use_webcam else -1

    # Configura Gravador de Saída
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Info: Processando: {source_path}")
    print(f"Info: Saída será salva em: {output_path}")
    print(f"Info: Total de frames: {total_frames}")

    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        
        # conf=conf_threshold define a confiança mínima para detectar
        results = model(frame, verbose=False, conf=conf_threshold, device=device_target)
        result = results[0]

        # Contadores do Frame Atual
        stats = {
            "helmet": 0,
            "no-helmet": 0,
            "person": 0,
            "vest": 0,
            "no-vest": 0
        }

        # Extrair dados
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            
            if cls_name in stats:
                stats[cls_name] += 1

        # Desenhar Bounding Boxes
        annotated_frame = result.plot()

        # Cria um retângulo preto semitransparente no canto superior esquerdo
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (350, 160), (0, 0, 0), -1)
        annotated_frame = cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0)

        # Escrever Estatísticas
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        white = (255, 255, 255)
        green = (0, 255, 0)
        red = (0, 0, 255)
        thickness = 2

        y_offset = 30
        cv2.putText(annotated_frame, f"Pessoas: {stats['person']}", (15, y_offset), font, font_scale, white, thickness)
        
        y_offset += 30
        cv2.putText(annotated_frame, f"Com Capacete: {stats['helmet']}", (15, y_offset), font, font_scale, green, thickness)
        
        y_offset += 30
        cv2.putText(annotated_frame, f"Com Colete: {stats['vest']}", (15, y_offset), font, font_scale, green, thickness)

        # Alertas de Segurança
        if stats['no-helmet'] > 0:
            y_offset += 35
            cv2.putText(annotated_frame, f"ALERTA: SEM CAPACETE ({stats['no-helmet']})", (15, y_offset), font, font_scale, red, thickness)
        
        if stats['no-vest'] > 0:
            y_offset += 30
            cv2.putText(annotated_frame, f"ALERTA: SEM COLETE ({stats['no-vest']})", (15, y_offset), font, font_scale, red, thickness)

        # Salvar frame processado
        out.write(annotated_frame)

        # Barra de progresso simples no terminal
        if frame_count % 30 == 0:
            print(f"Processado: {frame_count}/{total_frames} frames ({frame_count/total_frames:.1%})", end='\r')

    cap.release()
    out.release()
    print(f"\n[SUCESSO] Vídeo finalizado e salvo em: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detector de EPIs em Vídeo usando YOLOv8")
    
    parser.add_argument("--source", type=str, required=True, help="Caminho do vídeo de entrada (ex: video.mp4)")
    parser.add_argument("--model", type=str, default="weights/best.pt", help="Caminho do arquivo de pesos .pt")
    parser.add_argument("--output", type=str, default="output_video.mp4", help="Nome do arquivo de saída")
    parser.add_argument("--conf", type=float, default=0.5, help="Limiar de confiança (0.0 a 1.0)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu ou 0 para gpu")

    args = parser.parse_args()

    run_inference(args.source, args.model, args.output, args.conf, args.device)