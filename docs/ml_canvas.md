# ML Canvas — Telco Churn Prediction

Frame estruturado das decisões de ciclo de vida do modelo (formato Louis Dorard).

## 1. Proposta de valor

Reduzir o cancelamento de contratos em uma operadora de telecomunicações, identificando antecipadamente clientes com alto risco de churn para que o time de retenção possa ofertar incentivos direcionados (descontos, upgrades, atendimento prioritário).

## 2. Stakeholders

- **Time de retenção / Customer Success** — consumidor primário do score; usa para priorizar contatos.
- **Marketing** — segmenta campanhas de fidelidade.
- **Diretoria comercial** — patrocinador da iniciativa; mede ROI agregado.
- **Time de dados / engenharia de ML** — operadores e mantenedores do modelo.

## 3. Decisões / ações alimentadas pelo modelo

- **Lista priorizada** de contatos para o time de retenção (ordenada por probabilidade decrescente).
- **Filtros de campanha** no marketing (incluir clientes com score acima de X).
- **Dashboards executivos** mostrando concentração de risco por contrato/produto.

A predição **alimenta uma decisão humana**, nunca a substitui.

## 4. ML Task

- **Tipo**: classificação binária supervisionada.
- **Target**: `Churn` ∈ {Yes, No}, codificado para {1, 0}.
- **Modelo central**: rede neural MLP (PyTorch) — 2 hidden layers (128, 64), ReLU + dropout 0,3.
- **Baselines comparativos**: `DummyClassifier` (majoritária), regressão logística com `class_weight="balanced"`, Random Forest com 200 árvores.

## 5. Predições

- **Quando**: sob demanda via API REST (`POST /predict`) — inferência real-time.
- **Granularidade**: 1 cliente por chamada; lote via `POST /predict/batch` (até 1.000 clientes).
- **Frescor de dados**: features são lidas do CRM no momento da chamada.
- **SLO de latência**: p95 ≤ 100 ms (medido localmente: ~25 ms).

## 6. Features

- **Demográficas (4)**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
- **Serviço (10)**: `tenure`, `Contract`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
- **Financeiras (5)**: `MonthlyCharges`, `TotalCharges`, `PaperlessBilling`, `PaymentMethod`.
- **Engenharia adicionada**: `tenure_bin` (discretização de `tenure` em 4 faixas).
- **Identificador descartado**: `customerID`.
- **Total**: 19 features brutas → 49 colunas após one-hot encoding.

## 7. Coleta de dados (treino)

- **Fonte**: dataset público IBM Telco Customer Churn (slug Kaggle `blastchar/telco-customer-churn`).
- **Volumetria**: 7.043 clientes × 21 colunas.
- **Distribuição do target**: 73,46% No / 26,54% Yes.
- **Política**: dados não são re-coletados a cada deploy. Em produção, novos snapshots seriam treinados sobre as mesmas colunas extraídas do CRM real do cliente final.

## 8. Treino e atualização

- **Pipeline**: `make train` orquestra download → split estratificado 80/20 → grid search MLP (5-fold CV) → fit final → avaliação em 4 operating points → persistência de artefatos.
- **Cadência proposta**: re-treino mensal alinhado com o ciclo de cobrança.
- **Trigger automático**: PSI > 0,2 em qualquer feature, ou queda de F1 > 0,05 vs baseline (ver `monitoring.md`).
- **Versionamento**: cada run é rastreado no MLflow; artefatos versionados em `models/`.

## 9. Avaliação offline

- **Métricas técnicas reportadas**: ROC AUC, PR AUC, F1, precision, recall, accuracy.
- **Métrica priorizada**: F1 (balanceia precision/recall em problema desbalanceado).
- **Métrica de negócio (proxy)**: custo esperado `1·FP + 5·FN` (perder cliente custa ~5× mais que ofertar retenção a quem não sairia).
- **Threshold de produção**: 0,44 (ponto que maximiza F1 no test set holdout 20%).

## 10. Avaliação online e monitoramento

Detalhado em [`monitoring.md`](monitoring.md). Resumo das três camadas:

1. **Operacional** (Prometheus + Grafana): latência p50/p95/p99, error rate 5xx, throughput.
2. **Drift estatístico**: PSI por feature, distribuição da probabilidade de saída, reconciliação periódica com churn observado.
3. **Negócio**: taxa real de churn, conversão de campanhas, ROI agregado.
