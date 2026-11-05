import cv2
import numpy as np

def draw_counts_panel(frame: np.ndarray, counts: dict[str, int]) -> np.ndarray:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (350, 160), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    # Escrever Estatísticas
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    white = (255, 255, 255)
    green = (0, 255, 0)
    red = (0, 0, 255)
    thickness = 2

    y_offset = 30
    cv2.putText(frame, f"Pessoas: {counts['person']}", (15, y_offset), font, font_scale, white, thickness)
    
    y_offset += 30
    cv2.putText(frame, f"Com Capacete: {counts['helmet']}", (15, y_offset), font, font_scale, green, thickness)
    
    y_offset += 30
    cv2.putText(frame, f"Com Colete: {counts['vest']}", (15, y_offset), font, font_scale, green, thickness)

    # Alertas de Segurança
    if counts['no-helmet'] > 0:
        y_offset += 35
        cv2.putText(frame, f"ALERTA: SEM CAPACETE ({counts['no-helmet']})", (15, y_offset), font, font_scale, red, thickness)
    
    if counts['no-vest'] > 0:
        y_offset += 30
        cv2.putText(frame, f"ALERTA: SEM COLETE ({counts['no-vest']})", (15, y_offset), font, font_scale, red, thickness)

    return frame