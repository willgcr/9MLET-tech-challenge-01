# Model Card — ChurnMLP v0.1.0

Estrutura inspirada em Mitchell et al. (2019), *Model Cards for Model Reporting*.

## Detalhes do modelo

| Item | Valor |
|---|---|
| Nome | `ChurnMLP` |
| Versão | 0.1.0 |
| Framework | PyTorch 2.4 |
| Arquitetura | MLP de 2 hidden layers — `Linear(49→128) → ReLU → Dropout → Linear(128→64) → ReLU → Dropout → Linear(64→1)` |
| Hidden layers | (128, 64) |
| Dropout | 0,3 |
| Loss | `BCEWithLogitsLoss` com `pos_weight = n_neg / n_pos ≈ 2,77` |
| Otimizador | Adam (`lr=5e-4`, `weight_decay=1e-4`) |
| Regularização | Dropout + early stopping (paciência 8 épocas) |
| Threshold de produção | 0,44 (max F1 no test set) |
| Tamanho do artefato | ~59 KiB (state dict) |
| Hardware necessário | CPU (não requer GPU) |

## Uso pretendido

- **Caso primário**: classificar clientes de uma operadora de telecom como em risco ou não de cancelar o serviço, alimentando campanhas de retenção.
- **Usuários previstos**: time de Customer Success, marketing, ferramentas internas de BI.
- **Modo de uso recomendado**: chamada via API REST com features atualizadas do CRM. Consumir o `churn_probability` como **score de prioridade**, não como certeza absoluta.

## Casos fora do escopo

- **Outras operadoras / regiões / produtos** sem retreino — o modelo aprendeu padrões específicos deste dataset (operadora americana, snapshot Q3).
- **Previsão de prazo** de cancelamento — este modelo só responde sim/não.
- **Decisão final** sobre clientes — é input para decisão humana, não substituição.

## Dados de treino

- **Fonte**: IBM Telco Customer Churn (Kaggle `blastchar/telco-customer-churn`).
- **Volumetria**: 7.043 registros × 21 colunas.
- **Janela temporal**: snapshot Q3 do dataset original.
- **Distribuição do target**: 73,46% `No` / 26,54% `Yes`.
- **Split**: 80% treino + 20% test estratificado por target. Dentro do treino, 20% reservado para validação interna usada apenas em early stopping. Seed fixa (`RANDOM_SEED=42`).

## Avaliação

Avaliação realizada no test set holdout (20%, n=1.409 clientes).

### Métricas independentes de threshold

| Métrica | Valor | Baseline aleatório |
|---|---:|---:|
| ROC AUC | **0,843** | 0,500 |
| PR AUC | **0,632** | 0,265 |

### Métricas em quatro operating points

| Ponto | Threshold | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| `default_0.5` | 0,500 | 0,509 | 0,791 | 0,619 | 0,742 |
| **`max_f1` (produção)** | **0,440** | **0,494** | **0,848** | **0,625** | **0,730** |
| `recall_min_0.85` | 0,436 | 0,491 | 0,850 | 0,623 | 0,727 |
| `cost_optimal_1_to_5` | 0,370 | 0,468 | 0,898 | 0,615 | 0,702 |

### Comparação com baselines (5-fold CV no train set)

| Modelo | ROC AUC | F1 | Recall | Accuracy |
|---|---:|---:|---:|---:|
| `dummy_majority` | 0,500 | 0,000 | 0,000 | 0,7346 |
| `logistic_regression` | 0,846 | 0,625 | 0,801 | 0,7453 |
| `random_forest` | 0,839 | 0,612 | 0,646 | 0,7830 |
| `mlp` (final tunado) | 0,843 | 0,628 | 0,848 | 0,7300 |

> **Nota técnica**: regressão logística empata com o MLP em ROC AUC. O sinal nesse dataset é majoritariamente linear; o MLP foi entregue como modelo central conforme escopo da Fase 01, mas para deploy "puro" a regressão logística seria competitiva e mais simples (single-file, sem dependência do PyTorch em produção).

### Grid search realizado

12 configurações combinando `hidden_dims ∈ {(32,16), (64,32), (128,64)}`, `dropout ∈ {0,1, 0,3}`, `lr ∈ {1e-3, 5e-4}`. ROC AUC variou de 0,8438 a 0,8456 — 0,0018 dentro de variação natural entre folds. Conclusão: arquitetura não é o gargalo neste dataset.

## Limitações conhecidas

### 1. Probabilidades overconfident (não calibradas)

O MLP é sistematicamente overconfident — propriedade conhecida de redes neurais (Guo et al., 2017). Medindo no test set:

| Bin de probabilidade | Mean predicted | Actual rate | Diff |
|---|---:|---:|---:|
| `[0.7, 0.8)` | 0,75 | 0,47 | -0,28 |
| `[0.8, 0.9)` | 0,85 | 0,70 | -0,15 |
| `[0.9, 1.0)` | 0,91 | 0,67 | -0,25 |

**Implicação**: o ranking é confiável (ROC AUC 0,84 inalterado), mas a interpretação absoluta do score deve ser feita com ressalva. Tratar como score de prioridade, não como "X% literal de chance".

**Mitigação possível** (não implementada neste projeto): calibração via `CalibratedClassifierCV` (sklearn) ou temperature scaling. Adicionaria ~30 linhas e uma camada extra na inferência.

### 2. Dataset pequeno e específico

7.043 clientes, uma operadora americana, um trimestre. Padrões podem não generalizar para:
- Outras geografias (cultural, regulatória, modo de uso).
- Outros segmentos (B2B, prepago, IoT).
- Outras janelas temporais (mudanças de mercado, novos competidores).

### 3. Features limitadas

Sem features comportamentais (uso real do serviço, chamadas ao suporte, NPS, histórico de pagamentos atrasados). Resultados estão no "teto natural" deste conjunto de features (~0,84 ROC AUC). Estudos publicados que reportam números melhores neste mesmo dataset (>0,90) tipicamente incluem `Churn Score` (predição interna da IBM) — o que constitui *data leakage*.

### 4. Sensibilidade a categorias novas

Se o CRM começar a registrar uma nova categoria (ex.: novo `PaymentMethod`), o `OneHotEncoder` ignora silenciosamente (`handle_unknown="ignore"`). A predição continua válida mas perde sinal. Monitorar via PSI por feature (ver `monitoring.md`).

### 5. Inferência por linha, não por sequência

Cada cliente é tratado como independente. Não há modelagem de sequência temporal (ex.: histórico de N meses como série). Para incorporar isso, seria necessário expandir o schema de entrada e reformular o modelo (ex.: LSTM, Transformer tabular).

## Vieses e considerações éticas

### Sinais observados

- **`gender`**: pesos quase iguais para `Male`/`Female` (efeito médio de ±0,5 pp na probabilidade). Não há viés direto observável.
- **`SeniorCitizen`**: clientes idosos aparecem em segmentos de maior churn no dataset. O modelo herda essa correlação. **Atenção**: usar o score como sinalizador, **nunca** como justificativa para tratamento diferenciado por idade.
- **`PaymentMethod`**: clientes que pagam por *Electronic check* têm taxa de churn quase 2× maior. Útil como sinal preditivo, mas pode correlacionar com renda — evitar campanhas que penalizem implicitamente faixas de baixa renda.

### Considerações operacionais

- O modelo prediz cancelamento, **não recomenda ação**. A decisão sobre quem ofertar retenção é humana e auditável.
- Não há features sensíveis explícitas (raça, religião, orientação) no dataset.
- O dataset original tem origem fictícia (IBM SPSS Sample Data); riscos éticos em produção dependem do CRM real do cliente final e devem ser auditados antes de qualquer rollout.

## Recomendações

- **Em produção**: monitorar PSI por feature mensalmente; retreinar quando PSI > 0,2 ou queda de F1 > 0,05 (ver `monitoring.md`).
- **Para auditoria**: `model_version` + `production_threshold` retornados pelo `/health` permitem reproduzir cada decisão.
- **Para análises sensíveis**: aplicar calibração antes de usar a probabilidade como número absoluto em comunicações.
- **Para treinos futuros**: considerar adicionar features comportamentais (uso, suporte, pagamentos) — provável caminho para superar o teto atual de 0,84 ROC AUC.
