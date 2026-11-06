## Benchmark

- Modelo: `weights/best_v3.pt`
- Vídeo: `videos/EPI.mp4` (300 frames × 3 rodadas alternadas, aquecimento de 20)
- CPU: Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz | GPU: NVIDIA GeForce MX110

| Variante | Inferência média (ms) | p50 | p95 | Total/frame (ms) | FPS |
|---|---|---|---|---|---|
| PyTorch CPU | 180.7 | 152.8 | 236.5 | 183.8 | 5.4 |
| PyTorch GPU | 108.4 | 108.4 | 108.6 | 112.5 | 8.9 |
| ONNX Runtime CPU | 170.4 | 181.7 | 196.4 | 175.6 | 5.7 |
| OpenVINO CPU | 179.6 | 183.9 | 216.0 | 184.5 | 5.4 |

## Precisão (split `test`)

| Variante | mAP50 | mAP50-95 | helmet | no-helmet | no-vest | person | vest |
|---|---|---|---|---|---|---|---|
| PyTorch CPU | 0.790 | 0.417 | 0.902 | 0.657 | 0.654 | 0.859 | 0.879 |
| PyTorch GPU | 0.790 | 0.417 | 0.902 | 0.657 | 0.654 | 0.859 | 0.879 |
| ONNX Runtime CPU | 0.790 | 0.417 | 0.902 | 0.657 | 0.654 | 0.859 | 0.879 |
| OpenVINO CPU | 0.790 | 0.417 | 0.902 | 0.657 | 0.654 | 0.859 | 0.879 |
