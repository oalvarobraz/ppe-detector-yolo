## Benchmark

- Modelo: `weights/best_v3.pt`
- Vídeo: `videos/EPI.mp4` (300 frames × 3 rodadas alternadas, aquecimento de 20)
- CPU: Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz | GPU: NVIDIA GeForce MX110

| Variante | Inferência média (ms) | p50 | p95 | Total/frame (ms) | FPS |
|---|---|---|---|---|---|
| PyTorch CPU | 213.9 | 229.2 | 242.9 | 217.7 | 4.6 |
| PyTorch GPU | 108.5 | 108.5 | 108.7 | 113.7 | 8.8 |
| ONNX Runtime CPU | 258.8 | 276.0 | 321.0 | 265.3 | 3.8 |
| OpenVINO CPU | 206.8 | 197.9 | 243.2 | 212.3 | 4.7 |
| ONNX Runtime INT8 CPU | 321.1 | 282.4 | 453.2 | 327.0 | 3.1 |
| OpenVINO INT8 CPU | 108.4 | 106.5 | 123.6 | 114.3 | 8.7 |

## Precisão (split `test`)

| Variante | mAP50 | mAP50-95 | helmet | no-helmet | no-vest | person | vest |
|---|---|---|---|---|---|---|---|
| PyTorch CPU | 0.790 | 0.417 | 0.902 | 0.657 | 0.654 | 0.859 | 0.879 |
| PyTorch GPU | 0.790 | 0.417 | 0.902 | 0.657 | 0.654 | 0.859 | 0.879 |
| ONNX Runtime CPU | 0.790 | 0.417 | 0.902 | 0.657 | 0.654 | 0.859 | 0.879 |
| OpenVINO CPU | 0.790 | 0.417 | 0.902 | 0.657 | 0.654 | 0.859 | 0.879 |
| ONNX Runtime INT8 CPU | 0.790 | 0.410 | 0.904 | 0.680 | 0.632 | 0.859 | 0.874 |
| OpenVINO INT8 CPU | 0.789 | 0.418 | 0.900 | 0.670 | 0.649 | 0.857 | 0.869 |
