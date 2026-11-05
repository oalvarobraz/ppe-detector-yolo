## Benchmark

- Modelo: `weights/best_v2n.pt`
- Vídeo: `videos/EPI.mp4` (300 frames × 3 rodadas alternadas, aquecimento de 20)
- CPU: Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz | GPU: NVIDIA GeForce MX110

| Variante | Inferência média (ms) | p50 | p95 | Total/frame (ms) | FPS |
|---|---|---|---|---|---|
| PyTorch CPU | 67.0 | 55.0 | 87.5 | 70.3 | 14.2 |
| PyTorch GPU | 57.6 | 57.6 | 57.8 | 61.9 | 16.1 |
| ONNX Runtime CPU | 69.6 | 74.3 | 82.8 | 74.7 | 13.4 |
| OpenVINO CPU | 63.8 | 67.3 | 73.3 | 69.0 | 14.5 |

## Precisão (split `test`)

| Variante | mAP50 | mAP50-95 | helmet | no-helmet | no-vest | person | vest |
|---|---|---|---|---|---|---|---|
| PyTorch CPU | 0.744 | 0.375 | 0.889 | 0.457 | 0.634 | 0.895 | 0.845 |
| PyTorch GPU | 0.744 | 0.375 | 0.889 | 0.457 | 0.634 | 0.895 | 0.845 |
| ONNX Runtime CPU | 0.744 | 0.375 | 0.889 | 0.457 | 0.634 | 0.895 | 0.845 |
| OpenVINO CPU | 0.744 | 0.375 | 0.889 | 0.457 | 0.634 | 0.895 | 0.845 |
