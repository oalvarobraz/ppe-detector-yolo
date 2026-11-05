## Benchmark

- Modelo: `weights/best_v2.pt`
- Vídeo: `videos/EPI.mp4` (300 frames × 3 rodadas alternadas, aquecimento de 20)
- CPU: Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz | GPU: NVIDIA GeForce MX110

| Variante | Inferência média (ms) | p50 | p95 | Total/frame (ms) | FPS |
|---|---|---|---|---|---|
| PyTorch CPU | 191.0 | 220.0 | 241.7 | 194.0 | 5.2 |
| PyTorch GPU | 108.4 | 108.4 | 108.6 | 112.8 | 8.9 |
| ONNX Runtime CPU | 245.8 | 207.0 | 314.5 | 251.1 | 4.0 |
| OpenVINO CPU | 202.6 | 197.0 | 228.0 | 207.4 | 4.8 |

## Precisão (split `test`)

| Variante | mAP50 | mAP50-95 | helmet | no-helmet | no-vest | person | vest |
|---|---|---|---|---|---|---|---|
| PyTorch CPU | 0.776 | 0.394 | 0.916 | 0.645 | 0.677 | 0.870 | 0.774 |
| PyTorch GPU | 0.776 | 0.394 | 0.916 | 0.645 | 0.677 | 0.870 | 0.774 |
| ONNX Runtime CPU | 0.776 | 0.394 | 0.916 | 0.645 | 0.677 | 0.870 | 0.774 |
| OpenVINO CPU | 0.776 | 0.394 | 0.916 | 0.645 | 0.677 | 0.870 | 0.774 |
