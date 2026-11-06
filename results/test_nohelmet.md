## Teste extra: "sem capacete"

- Conjunto: `datasets/test_nohelmet`, gerado por `src/scripts/build_test_nohelmet.py`
- Fonte: Hard Hat Workers v5 (Roboflow, domínio público), **sem nenhuma imagem usada no treino** (hash perceptual, distância ≤ 8 bits = repetida)
- **1.378 imagens:** 378 com cabeça sem capacete e 1.000 só com capacete
- **Caixas:** `no-helmet` = 1.995 (o teste original tem 24); `helmet` = 3.815
- Só `helmet` e `no-helmet` estão anotados nessas imagens
- Avaliação com imgsz 640 e `rect=False`

### AP por classe

| Modelo | AP50 no-helmet | AP50-95 no-helmet | AP50 helmet | AP50-95 helmet |
|---|---:|---:|---:|---:|
| `best.pt` (v8n original) | 0.278 | 0.055 | 0.568 | 0.121 |
| `best_v2.pt` (v8s) | 0.558 | 0.130 | 0.689 | 0.165 |
| `best_v3.pt` (v8s + dados extras) ⚠️ | 0.944 | 0.571 | 0.752 | 0.246 |

### no-helmet por limiar (`src/scripts/tune_threshold.py`)

| Limiar | best.pt P / R | best_v2.pt P / R | best_v3.pt P / R ⚠️ |
|---:|---:|---:|---:|
| 0.20 | 0.48 / 0.32 | 0.68 / 0.55 | 0.79 / 0.96 |
| 0.30 | 0.53 / 0.23 | 0.75 / 0.50 | 0.84 / 0.95 |
| 0.40 | 0.56 / 0.14 | 0.78 / 0.44 | 0.87 / 0.93 |
| 0.50 | 0.59 / 0.06 | 0.81 / 0.37 | 0.91 / 0.92 |

### Como interpretar

- **`best.pt` vs `best_v2.pt` é uma comparação justa**, porque nenhum dos dois viu essa coleção no treino. Com 1.995 caixas, fica confirmado que o modelo maior praticamente dobra o AP50 de `no-helmet` (0.28 → 0.56).
- ⚠️ **O `best_v3.pt` está favorecido.** As imagens são inéditas, mas vêm da mesma coleção (mesmo estilo de foto e de anotação) das 920 imagens extras do treino dele. O salto do AP50-95 de `no-helmet` (0.13 → 0.57) indica que ele aprendeu também o padrão de caixas desses anotadores. **Este conjunto não mede o desempenho do v3 em câmeras de obra reais.**
- Para avaliar o v3 no uso real, falta um teste do domínio de aplicação: imagens de obras reais, anotadas à mão.
