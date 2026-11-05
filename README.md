# 🦺 Safety-Vision: Detector de EPIs com YOLOv8

> **Modelo:** YOLOv8n (Object Detection) · **mAP50 (validação):** 0.82
> **Formatos:** PyTorch · ONNX · OpenVINO — com benchmark de velocidade e precisão entre eles

Este projeto de Visão Computacional usa **Deep Learning** para monitorar automaticamente a segurança em canteiros de obras, identificando o uso de Equipamentos de Proteção Individual (EPIs), como capacetes e coletes, em vídeo ou webcam.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green?style=for-the-badge)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white)
![OpenVINO](https://img.shields.io/badge/OpenVINO-Intel-0071C5?style=for-the-badge&logo=intel&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-CV-red?style=for-the-badge&logo=opencv&logoColor=white)

---

## 📸 Demonstração

<p align="center">
  <img src="assets/demo.gif" alt="Demonstração do Sistema" width="300">
</p>

*Detecção de pessoas, capacetes e coletes em vídeo de canteiro de obras, com painel de contagem e alertas.*

---

## 🗂️ Estrutura do Projeto

```text
ppe-detector-yolo/
├── src/
│   ├── ppe_detector/            # Pacote principal (importável)
│   │   ├── detector.py          # PPEDetector: carrega .pt/.onnx/OpenVINO e conta as detecções
│   │   ├── overlay.py           # Painel de estatísticas e alertas desenhado no frame
│   │   └── video.py             # Loop de vídeo/webcam (gravação, janela, Ctrl+C seguro)
│   └── scripts/                 # Linha de comando
│       ├── infer.py             # Inferência em vídeo ou webcam
│       ├── export_model.py      # Exporta best.pt -> ONNX ou OpenVINO
│       ├── download_dataset.py  # Baixa o dataset do Roboflow (usa o .env)
│       └── benchmark.py         # Compara velocidade e mAP entre os formatos
├── notebooks/
│   └── exploratory_training.ipynb   # Treinamento original (Google Colab)
├── weights/
│   ├── best.pt                  # Pesos treinados (PyTorch)
│   ├── best.onnx                # Gerado pelo export_model.py
│   └── best_openvino_model/     # Gerado pelo export_model.py --format openvino
├── results/benchmark.md         # Último relatório gerado pelo benchmark.py
├── assets/                      # Imagens e GIF da documentação
└── pyproject.toml               # Dependências e configuração do pacote
```

---

## 🧠 Sobre o Projeto

A segurança em canteiros de obras é **crítica** e regulamentada por normas trabalhistas. Este projeto automatiza o monitoramento de conformidade com EPIs obrigatórios.

### 🔹 Funcionalidades

- ✅ **Detecção de pessoas, capacetes e coletes** em vídeo ou webcam
- ⚠️ **Alertas visuais** para ausência de EPIs (`no-helmet` / `no-vest`)
- 📊 **Painel de contagem** por frame
- 🎥 **Vídeo de saída anotado**, salvo corretamente mesmo ao interromper com Ctrl+C
- ⚡ **Exportação para ONNX e OpenVINO**, com benchmark de velocidade e precisão

### 🔹 Classes Detectadas

| Classe | Descrição |
|--------|-----------|
| 👷 `person` | Pessoa detectada |
| ⛑️ `helmet` | Capacete presente |
| 🦺 `vest` | Colete presente |
| 🚫 `no-helmet` | Ausência de capacete |
| 🚫 `no-vest` | Ausência de colete |

---

## ⚙️ Instalação

### 1️⃣ Clone o repositório e crie um ambiente virtual

```bash
git clone https://github.com/seu-usuario/ppe-detector-yolo.git
cd ppe-detector-yolo
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows
```

### 2️⃣ Instale o PyTorch para o seu hardware

O PyTorch é instalado separadamente porque a versão certa depende da sua GPU. Escolha **uma** opção:

```bash
# GPU NVIDIA (CUDA 12.6 — inclui GPUs antigas, como Maxwell/MX110)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Apenas CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> As versões CUDA mais novas do PyTorch deixaram de suportar GPUs Maxwell (compute capability 5.0). Com elas, `torch.cuda.is_available()` até retorna `True`, mas a inferência falha com *"no kernel image is available"*.

### 3️⃣ Instale o projeto

```bash
pip install -e ".[all]"
```

| Extra | O que instala |
|-------|---------------|
| *(nenhum)* | Inferência com o `.pt` (ultralytics, OpenCV) |
| `export` | ONNX, ONNX Runtime e OpenVINO |
| `dataset` | Roboflow e python-dotenv (download do dataset) |
| `all` | Tudo acima |

---

## 🚀 Como Usar

### 🎥 Inferência em vídeo ou webcam

```bash
# Vídeo, salvando o resultado
python src/scripts/infer.py --source videos/obra.mp4 --output resultado.mp4

# Webcam, mostrando numa janela (tecla 'q' ou Ctrl+C para sair)
python src/scripts/infer.py --source 0 --show

# Com GPU NVIDIA e modelo ONNX
python src/scripts/infer.py --source videos/obra.mp4 --model weights/best.onnx --device 0
```

| Argumento | Descrição | Padrão |
|-----------|-----------|--------|
| `--source` | Arquivo de vídeo ou índice da webcam (`0`, `1`, ...) | *(obrigatório)* |
| `--model` | Pesos: `.pt`, `.onnx` ou pasta `_openvino_model` | `weights/best.pt` |
| `--output` | Arquivo de vídeo de saída | `output_video.mp4` |
| `--conf` | Limiar de confiança (0.0 a 1.0) | `0.5` |
| `--device` | `cpu`, `0` (GPU NVIDIA) ou `intel:cpu` (OpenVINO) | `cpu` |
| `--show` | Mostra o resultado numa janela | desligado |

### 📦 Exportação

```bash
python src/scripts/export_model.py                      # -> weights/best.onnx
python src/scripts/export_model.py --format openvino    # -> weights/best_openvino_model/
```

### ⏱️ Benchmark

O dataset de validação vem do Roboflow. Crie um arquivo `.env` na raiz com a sua chave (o arquivo é ignorado pelo git):

```bash
ROBOFLOW_API_KEY=sua_chave_aqui
```

```bash
python src/scripts/download_dataset.py
python src/scripts/benchmark.py --video videos/obra.mp4 --data datasets/construction-safety/data.yaml
```

Sem `--data`, o benchmark mede só a velocidade. O relatório é salvo em `results/benchmark.md`.

---

## 📊 Resultados

Medido com `src/scripts/benchmark.py` num notebook **Intel Core i7-8565U** (4 núcleos, 8ª geração) com **GeForce MX110** (2 GB), perfil de energia em *performance*. Foram 300 frames de vídeo × 3 rodadas alternadas entre as variantes, após 20 frames de aquecimento.

### ⚡ Velocidade (entrada 640×640, batch 1)

| Variante | Inferência p50 (ms) | p95 (ms) | Total/frame (ms) | FPS |
|----------|--------------------:|---------:|-----------------:|----:|
| PyTorch — GPU MX110 | **51.7** | **51.9** | 56.5 | **17.7** |
| OpenVINO — CPU | 63.0 | 84.6 | 71.6 | 14.0 |
| ONNX Runtime — CPU | 71.5 | 110.9 | 84.6 | 11.8 |
| PyTorch — CPU | 78.8 | 96.8 | 77.9 | 12.8 |

*Total/frame inclui pré-processamento (redimensionamento) e pós-processamento (NMS).*

- **GPU:** a mais rápida e a mais estável (p50 ≈ p95), mesmo sendo uma GPU de entrada.
- **CPU:** o **OpenVINO** foi o mais rápido, cerca de 20% à frente do PyTorch e 12% à frente do ONNX Runtime, como esperado num motor otimizado para hardware Intel.
- **ONNX Runtime vs PyTorch na CPU:** resultado **inconclusivo** nesta máquina. Numa execução anterior, o PyTorch CPU marcou 50 ms (p50) contra 70 ms do ONNX. O PyTorch CPU foi a variante mais sensível a temperatura e carga do sistema, enquanto ONNX (70–72 ms) e GPU (52 ms) se repetiram entre execuções.

### 🎯 Precisão (split de validação, 119 imagens)

| Variante | mAP50 | mAP50-95 | helmet | no-helmet | no-vest | person | vest |
|----------|------:|---------:|-------:|----------:|--------:|-------:|-----:|
| PyTorch (CPU e GPU) | 0.816 | 0.467 | 0.933 | 0.480 | 0.827 | 0.938 | 0.903 |
| ONNX Runtime | 0.816 | 0.467 | 0.933 | 0.480 | 0.827 | 0.938 | 0.903 |
| OpenVINO | 0.816 | 0.467 | 0.933 | 0.480 | 0.827 | 0.938 | 0.903 |

**As exportações não perderam precisão:** as quatro variantes têm resultados idênticos. Para a mesma entrada, as saídas brutas do PyTorch e do ONNX diferem em menos de 0.001.

> Todas as variantes são validadas com imagem quadrada 640×640 (`rect=False`), porque os modelos exportados têm entrada fixa. Validando o `.pt` com o padrão do ultralytics (`rect=True`, padding mínimo por imagem), o mAP50 sobe para 0.841 e o de `no-helmet` para 0.609. Essa diferença vem do pré-processamento, não do modelo, e é amplificada pelas poucas amostras de `no-helmet`.

### 🔁 Retreino: YOLOv8n vs YOLOv8s (split de teste, 90 imagens)

Os retreinos foram feitos no Google Colab com `src/scripts/train.py` (100 épocas com `patience=20`; os dois pararam cedo). Todos os modelos foram avaliados no mesmo split de **teste**, nunca visto no treino.

| Modelo | mAP50 | mAP50-95 | helmet | **no-helmet** | no-vest | person | vest | GPU p50 | OpenVINO p50 |
|--------|------:|---------:|-------:|--------------:|--------:|-------:|-----:|--------:|-------------:|
| `best.pt` — v8n original (30 épocas) | 0.751 | **0.401** | 0.900 | 0.482 | 0.667 | 0.856 | **0.847** | **52 ms** | **63 ms** |
| v8n retreinado (melhor época: 19) | 0.744 | 0.375 | 0.889 | 0.457 | 0.634 | **0.895** | 0.845 | 58 ms | 67 ms |
| `best_v2.pt` — v8s (melhor época: 23) | **0.776** | 0.394 | **0.916** | **0.645** | **0.677** | 0.870 | 0.774 | 108 ms | 197 ms |

- **Retreinar o v8n não trouxe ganho.** A melhora em `no-helmet` (0.48 → 0.65) veio do **modelo maior**, ao custo de ser 2–3× mais lento.
- **A validação durante o treino superestima o modelo.** O v8n retreinado marcou mAP50 de 0.88 na validação (119 imagens, usadas para escolher o checkpoint) e 0.744 no teste.
- Relatórios completos: [`results/benchmark_v2.md`](results/benchmark_v2.md) e [`results/benchmark_v2n.md`](results/benchmark_v2n.md).

---

## 🧪 Treinamento

O modelo foi treinado no Google Colab (GPU T4) com o notebook `notebooks/exploratory_training.ipynb`:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=30, imgsz=640, batch=16, patience=10, cache=True)
```

### 🔹 Dataset

**Fonte:** [Roboflow 100 — Construction Safety](https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb) (versão 2, licença CC BY 4.0)

| Split | Imagens | helmet | no-helmet | no-vest | person | vest |
|-------|---------|--------|-----------|---------|--------|------|
| Treino | 997 | 1443 | **73** | 695 | 2175 | 1004 |
| Validação | 119 | 148 | **10** | 86 | 222 | 131 |
| Teste | 90 | 143 | **21** | 59 | 190 | 121 |

Total: **1.206 imagens** (83% treino / 10% validação / 7% teste).

> ⚠️ **O ponto fraco do modelo é o desbalanceamento:** há apenas **73** exemplos de `no-helmet` no treino, contra **1443** de `helmet` (cerca de 20×). Como só 10 caixas de `no-helmet` existem na validação, o mAP dessa classe também é instável: errar uma ou duas caixas muda bastante o número.

---

## 🛣️ Roadmap

- [x] Exportação para **ONNX** e **OpenVINO**, com benchmark de velocidade e precisão
- [ ] **Balanceamento do dataset:** mais imagens de `no-helmet` / `no-vest` e augmentation nas classes minoritárias
- [x] **Retreino** com modelo maior (YOLOv8s): `no-helmet` 0.48 → 0.65 no teste, porém 2–3× mais lento
- [ ] Adaptar ao contexto brasileiro (ex.: cinto/arnês de altura, que hoje é confundido com `no-vest`)
- [ ] **Quantização INT8** (OpenVINO/ONNX) e medição do trade-off velocidade × precisão
- [ ] **Zonas de perigo:** polígonos definidos pelo usuário com alertas específicos
- [ ] **Sistema de alertas:** log de violações com timestamp e captura de tela

---

## ✅ Principais Aprendizados

- **Formato não é sinônimo de velocidade:** exportar para ONNX não garantiu ganho nesta CPU Intel, porque o PyTorch já usa bibliotecas otimizadas (oneDNN). O ganho veio do **motor certo para o hardware** (OpenVINO em CPU Intel).
- **Compare de igual para igual:** a primeira medição de mAP indicou queda no ONNX, mas era diferença de **pré-processamento** (o `.pt` valida com `rect=True` por padrão). Com a mesma entrada, as saídas do PyTorch e do ONNX diferem em menos de 0.001.
- **Benchmark em notebook é ruidoso:** temperatura (82 °C), perfil de energia e outros programas mudam os tempos. Aquecimento, rodadas alternadas e percentis (p50/p95) tornam a comparação confiável.
- **Desbalanceamento de classes** afeta justamente o que mais importa num sistema de segurança: detectar a *falta* de EPI.

---

## 📌 Autor

Desenvolvido por **Álvaro Braz Cunha**.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/alvaro-braz-cunha)

Projeto desenvolvido para fins de **estudo, pesquisa e portfólio profissional em Visão Computacional e Deep Learning**.

---

## 🙏 Agradecimentos

- **Roboflow Universe:** dataset Construction Safety
- **Ultralytics:** framework YOLOv8
- **Google Colab:** infraestrutura de treinamento com GPU gratuita
