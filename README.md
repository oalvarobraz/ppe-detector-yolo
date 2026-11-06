# 🦺 Safety-Vision: Detector de EPIs com YOLOv8

> **Modelo:** YOLOv8s (`best_v3.pt`) · **mAP50 (teste):** 0.79 · **Recall de "sem capacete":** 0.04 → 0.67
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
│       ├── export_model.py      # Exporta .pt -> ONNX ou OpenVINO
│       ├── benchmark.py         # Compara velocidade e mAP entre os formatos
│       ├── tune_threshold.py    # Precisão/recall por limiar de confiança (escolha do limiar)
│       ├── build_test_nohelmet.py   # Teste extra de "sem capacete", sem imagens do treino
│       ├── train.py             # Treino (pensado para o Google Colab)
│       ├── download_dataset.py  # Baixa o dataset do Roboflow (usa o .env)
│       ├── download_hardhat.py  # Baixa as imagens "sem capacete" do Hard Hat Detection
│       ├── convert_hardhat.py   # Converte o Hard Hat (FiftyOne) para YOLO
│       └── build_merged_dataset.py  # Junta os datasets, com pseudo-rótulos
├── notebooks/
│   └── exploratory_training.ipynb   # Treinamento original (Google Colab)
├── weights/
│   ├── best_v3.pt               # Modelo atual: YOLOv8s + dados extras (padrão do infer.py)
│   ├── best_v2.pt               # YOLOv8s, só o dataset original
│   ├── best.pt                  # YOLOv8n original
│   └── *.onnx, *_openvino_model/    # Gerados pelo export_model.py (fora do git)
├── results/                     # Relatórios do benchmark.py de cada modelo
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

# Com GPU NVIDIA
python src/scripts/infer.py --source videos/obra.mp4 --device 0

# Na CPU com OpenVINO (o mais rápido em CPU Intel)
python src/scripts/infer.py --source videos/obra.mp4 --model weights/best_v3_openvino_model --device intel:cpu
```

| Argumento | Descrição | Padrão |
|-----------|-----------|--------|
| `--source` | Arquivo de vídeo ou índice da webcam (`0`, `1`, ...) | *(obrigatório)* |
| `--model` | Pesos: `.pt`, `.onnx` ou pasta `_openvino_model` | `weights/best_v3.pt` |
| `--output` | Arquivo de vídeo de saída | `output_video.mp4` |
| `--conf` | Limiar de confiança (0.0 a 1.0) | `0.5` |
| `--class-conf` | Limiares por classe (`classe=valor`); sem valores, usa `--conf` em tudo | `no-helmet=0.3` |
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

### 🔁 Retreino e mais dados (split de teste, 90 imagens)

Os retreinos foram feitos no Google Colab com `src/scripts/train.py` (100 épocas com `patience=20`; todos pararam antes). Todos os modelos foram avaliados no mesmo split de **teste**, nunca visto no treino.

| Modelo | mAP50 | mAP50-95 | helmet | **no-helmet** | no-vest | person | vest | GPU p50 |
|--------|------:|---------:|-------:|--------------:|--------:|-------:|-----:|--------:|
| `best.pt` — v8n original (30 épocas) | 0.751 | 0.401 | 0.900 | 0.482 | 0.667 | 0.856 | 0.847 | **52 ms** |
| v8n retreinado (melhor época: 19) | 0.744 | 0.375 | 0.889 | 0.457 | 0.634 | **0.895** | 0.845 | 58 ms |
| `best_v2.pt` — v8s (melhor época: 23) | 0.776 | 0.394 | **0.916** | 0.645 | **0.677** | 0.870 | 0.774 | 108 ms |
| **`best_v3.pt` — v8s + dados extras** (melhor época: 40) | **0.790** | **0.417** | 0.902 | **0.657** | 0.654 | 0.859 | **0.879** | 108 ms |

- **Retreinar o v8n não trouxe ganho.** A primeira melhora em `no-helmet` (v2) veio do **modelo maior**, ao custo de ser cerca de 2× mais lento.
- **Mais dados (v3) melhoraram o modelo no geral:** o melhor mAP50 e mAP50-95, e o `vest`, que tinha caído no v2, se recuperou (0.77 → 0.88).
- **A validação durante o treino superestima o modelo.** O v8n retreinado marcou mAP50 de 0.88 na validação (119 imagens, usadas para escolher o checkpoint) e 0.744 no teste.
- Relatórios completos: [`results/benchmark_v2.md`](results/benchmark_v2.md), [`results/benchmark_v2n.md`](results/benchmark_v2n.md) e [`results/benchmark_v3.md`](results/benchmark_v3.md).

### 🚨 O que importa no uso: o alerta "sem capacete" dispara?

O mAP resume todos os limiares de confiança, mas o detector usa **um** limiar. Medindo `no-helmet` no teste (24 casos reais) com `src/scripts/tune_threshold.py`:

| Modelo | Limiar | Precisão | **Recall** | Encontrados | Alarmes falsos |
|--------|-------:|---------:|-----------:|------------:|---------------:|
| `best.pt` (original) | 0.5 | 1.00 | **0.04** | 1 de 24 | 0 |
| `best_v2.pt` | 0.5 | 0.75 | 0.25 | 6 de 24 | 2 |
| `best_v3.pt` | 0.5 | 0.85 | 0.46 | 11 de 24 | 2 |
| **`best_v3.pt`** (padrão do `infer.py`) | **0.3** | 0.67 | **0.67** | **16 de 24** | 8 |

- **O modelo original praticamente nunca disparava o alerta** de "sem capacete": encontrava 1 caso em 24. O mAP de 0.48 escondia isso.
- O `infer.py` usa **limiar 0.3 só para `no-helmet`** (`--class-conf`) e 0.5 para as outras classes. Num sistema de segurança, um alarme falso custa menos que deixar passar alguém sem capacete.
- Com só 24 casos no teste, esses números têm bastante incerteza.

### 🧪 Teste extra de "sem capacete" (1.995 casos)

Para reduzir essa incerteza, `src/scripts/build_test_nohelmet.py` monta um segundo conjunto de teste com **1.378 imagens inéditas** do Hard Hat Workers. As imagens repetidas do treino foram removidas por **hash perceptual**. O conjunto tem 378 imagens com cabeças sem capacete e 1.000 só com capacete, somando **1.995 casos de `no-helmet`** (contra 24 no teste original).

| Modelo | AP50 no-helmet | Recall (limiar 0.3) | Precisão (limiar 0.3) |
|--------|---------------:|--------------------:|----------------------:|
| `best.pt` | 0.278 | 0.23 | 0.53 |
| `best_v2.pt` | 0.558 | 0.50 | 0.75 |
| `best_v3.pt` ⚠️ | 0.944 | 0.95 | 0.84 |

- **Confirma, agora com uma amostra grande, que o modelo maior dobra o AP50 de `no-helmet`** (original → v2). Essa comparação é justa, porque nenhum dos dois viu essa coleção.
- ⚠️ **O v3 está favorecido neste conjunto.** As imagens são inéditas, mas vêm da mesma coleção que as imagens extras do treino dele, com o mesmo estilo de foto e de anotação. O número dele **não** representa o desempenho em câmeras de obra reais.
- Detalhes em [`results/test_nohelmet.md`](results/test_nohelmet.md).

---

## 🧪 Treinamento

Os treinos rodam no Google Colab (GPU T4). A GPU local, uma MX110 de 2 GB, levaria mais de 20 horas.

```python
# No Colab
!git clone https://github.com/oalvarobraz/ppe-detector-yolo.git
%cd ppe-detector-yolo
!pip install -q -e ".[dataset]"
!python src/scripts/train.py --data datasets/merged/data.yaml --model yolov8s.pt --name ppe_v3 \
    --project /content/drive/MyDrive/ppe_runs     # salva no Drive, caso a sessão caia
```

Ao final, o `train.py` avalia o melhor checkpoint no split de **teste** e imprime o mAP por classe. O modelo original (`best.pt`) foi treinado com o notebook `notebooks/exploratory_training.ipynb` (YOLOv8n, 30 épocas).

### 🔹 Dataset original

**Fonte:** [Roboflow 100 — Construction Safety](https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb) (versão 2, licença CC BY 4.0)

| Split | Imagens | helmet | no-helmet | no-vest | person | vest |
|-------|---------|--------|-----------|---------|--------|------|
| Treino | 997 | 2116 | **94** | 741 | 2362 | 1073 |
| Validação | 119 | 232 | **11** | 90 | 241 | 141 |
| Teste | 90 | 195 | **24** | 61 | 214 | 129 |

Total: **1.206 imagens** (83% treino / 10% validação / 7% teste).

> ⚠️ **O ponto fraco do dataset é o desbalanceamento:** há apenas **94** exemplos de `no-helmet` no treino, contra **2116** de `helmet` (cerca de 22×).

### 🔹 Dataset ampliado (`best_v3.pt`)

Para corrigir o desbalanceamento, o treino recebeu as **920 imagens com "cabeça sem capacete"** do [Hard Hat Detection](https://huggingface.co/datasets/Voxel51/hard-hat-detection) (licença CC0):

```bash
python src/scripts/download_dataset.py       # dataset original (Roboflow)
python src/scripts/download_hardhat.py       # 920 imagens com a classe "head"
python src/scripts/convert_hardhat.py        # FiftyOne -> YOLO (head -> no-helmet)
python src/scripts/build_merged_dataset.py   # junta tudo em datasets/merged
```

- Esse dataset só anota capacete e cabeça. Para que coletes e pessoas não virassem "fundo", **`person`, `vest` e `no-vest` foram completados com pseudo-rótulos** do `best_v2.pt` (confiança ≥ 0.5).
- A **validação e o teste continuam sendo os originais**, para a comparação entre modelos ser justa.
- Resultado no treino: **1.917 imagens**, e `no-helmet` passou de 94 para 5.879 caixas.

---

## 🛣️ Roadmap

- [x] Exportação para **ONNX** e **OpenVINO**, com benchmark de velocidade e precisão
- [x] **Retreino** com modelo maior (YOLOv8s): `no-helmet` 0.48 → 0.65 no teste, porém cerca de 2× mais lento
- [x] **Mais dados de `no-helmet`** (Hard Hat Detection + pseudo-rótulos): recall do alerta 0.04 → 0.67
- [x] **Limiar de confiança por classe** (`no-helmet=0.3`), escolhido com `tune_threshold.py`
- [x] Teste extra de `no-helmet` com 1.995 casos inéditos (deduplicados por hash perceptual)
- [ ] **Teste do domínio de aplicação:** 200–300 imagens de obras reais anotadas à mão, para avaliar o v3 sem o viés da coleção de treino
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
- **mAP não é a métrica do usuário:** o modelo original tinha mAP50 de 0.48 em `no-helmet`, mas, no limiar usado na prática, encontrava 1 caso em 24. Precisão e recall no limiar real mostram o que o sistema faz.
- **Dados ganham de arquitetura:** o modelo maior ajudou, mas o maior salto veio de dados da classe rara, com pseudo-rótulos para não "apagar" as outras classes.

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
