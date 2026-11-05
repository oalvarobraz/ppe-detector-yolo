## Benchmark

- Vídeo: `videos/EPI.mp4` (300 frames × 3 rodadas alternadas, aquecimento de 20)
- CPU: Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz | GPU: NVIDIA GeForce MX110

| Variante | Inferência média (ms) | p50 | p95 | Total/frame (ms) | FPS |
|---|---|---|---|---|---|
| PyTorch CPU | 74.3 | 78.8 | 96.8 | 77.9 | 12.8 |
| PyTorch GPU | 51.7 | 51.7 | 51.9 | 56.5 | 17.7 |
| ONNX Runtime CPU | 79.1 | 71.5 | 110.9 | 84.6 | 11.8 |
| OpenVINO CPU | 66.5 | 63.0 | 84.6 | 71.6 | 14.0 |

## Precisão (split `val`)

| Variante | mAP50 | mAP50-95 | helmet | no-helmet | no-vest | person | vest |
|---|---|---|---|---|---|---|---|
| PyTorch CPU | 0.816 | 0.467 | 0.933 | 0.480 | 0.827 | 0.938 | 0.903 |
| PyTorch GPU | 0.816 | 0.467 | 0.933 | 0.480 | 0.827 | 0.938 | 0.903 |
| ONNX Runtime CPU | 0.816 | 0.467 | 0.933 | 0.480 | 0.827 | 0.938 | 0.903 |
| OpenVINO CPU | 0.816 | 0.467 | 0.933 | 0.480 | 0.827 | 0.938 | 0.903 |
