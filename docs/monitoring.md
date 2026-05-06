# Plano de monitoramento

Detectar degradação **operacional** (infra) e **estatística** (modelo) antes que afete a decisão de negócio.

## Camada 1 — operacional (infra)

Coletada via **Prometheus**, exibida em **Grafana**. Em produção, expor com `prometheus-fastapi-instrumentator` (não incluso neste escopo, padrão da indústria).

| Métrica | Tipo | Alerta sugerido |
|---|---|---|
| `http_request_duration_seconds` (p50, p95, p99) | histogram | p95 > 200 ms por 5 min → warning; >500 ms → critical |
| `http_requests_total{status="5xx"}` | counter | error rate > 1% por 5 min → critical |
| `http_requests_total` (RPS total) | counter | crescimento súbito >3× → investigar |
| `process_resident_memory_bytes` | gauge | >450 MB sustentado → escalar |
| `python_gc_collections_total` | counter | spikes prolongados → memory leak |
| `up` (Prometheus default) | gauge | up=0 por >2 min → critical |

A API atual já loga em JSON cada request com `latency_ms`, então qualquer log shipper (Loki, ELK, CloudWatch) consegue extrair as mesmas métricas sem instrumentação adicional.

## Camada 2 — estatística do modelo (drift)

Coletada offline, via job diário, sobre os logs da API + dados do CRM.

| Métrica | Cálculo | Alerta |
|---|---|---|
| **PSI** (Population Stability Index) por feature | comparar distribuição na janela atual (7 dias) vs baseline (treino) | PSI > 0,1 → warning; > 0,2 → trigger de retraining |
| **Distribuição da probabilidade de saída** | histograma da saída do modelo na janela | shift > 10% no quantil 50 ou 90 → investigar |
| **Reconciliação com churn observado** | comparar predições do mês N com cancelamentos reais do mês N | diff absoluta na taxa agregada > 5 pp → retraining |
| **Volume por categoria** | counts de cada `Contract`, `PaymentMethod`, etc. | nova categoria nunca vista no treino → retraining |

### O que é PSI?

Population Stability Index mede o quanto a distribuição de uma feature mudou:

```
PSI = Σ (p_atual − p_base) · ln(p_atual / p_base)
```

Para cada bin da feature, calcula a diferença em proporção. PSI < 0,1 = estável, 0,1–0,2 = mudança moderada, > 0,2 = mudança significativa.

## Camada 3 — negócio

| Métrica | Origem | Cadência |
|---|---|---|
| **Taxa real de churn** (segmentada por contrato/produto) | sistema de billing | mensal |
| **Conversão de campanhas de retenção** (% de clientes contatados que ficam) | CRM | semanal |
| **ROI agregado** (CLTV salvo − custo da campanha) | financeiro | trimestral |

## Playbook de resposta

### Latência p95 > 500 ms

1. Verificar gráfico de RPS — pico de tráfego?
2. Verificar utilização CPU/RAM da instância.
3. Se CPU alta sustentada → escalar horizontalmente (mais réplicas).
4. Se RAM alta → investigar memory leak via profiling.
5. Se nenhum dos anteriores → instrumentar request lento (header `X-Process-Time-Ms` já dá baseline) e procurar gargalo na pipeline.

### Error rate > 1% (5xx)

1. Olhar logs JSON estruturados → buscar `level="ERROR"` recentes.
2. Padrão mais comum: novo formato de payload do CRM (faltando campos / categoria nova) → coordenar com time consumidor.
3. Se erro vem de `predictor.predict_dataframe`, verificar artefatos em `models/` e relatar `model_version` retornado por `/health`.

### PSI > 0,2 em alguma feature

1. Identificar a feature.
2. Validar com time de produto: mudança de plano? nova categoria? mudança de operação?
3. Se mudança esperada → retreinar com janela atualizada (`make train` em snapshot novo).
4. Se mudança inesperada → coordenar com time fonte do dado e investigar antes de retreinar.

### Reconciliação com churn observado falha

1. Validar que predições foram comparadas contra a janela correta (a predição é feita para um período futuro, não corrente).
2. Se a diferença for sistemática (modelo subestima ou superestima sempre) → recalibrar (Platt scaling) ou retreinar.
3. Se a diferença for ruído aleatório → aumentar tamanho da janela de comparação.

## Cadência de retraining

- **Mensal** (planejado): retreino com janela rolling de 12 meses.
- **Disparado**: PSI > 0,2 em qualquer feature, ou queda de F1 > 0,05 vs baseline.
- **Manual**: a qualquer momento via `make train`.

Cada novo treino vira uma nova run no MLflow, comparável às anteriores. Se o novo modelo for inferior, o anterior continua em produção (rollback é só redeployar com os artefatos antigos).

## Tooling sugerido

- **Prometheus + Grafana** — métricas operacionais (mencionados na disciplina de APIs do curso).
- **MLflow** (já implementado) — tracking de runs, comparação de versões.
- **Job scheduler** (cron / Airflow / Prefect) — drift e reconciliação periódicos.
- **Logs estruturados em JSON** (já implementado) — qualquer plataforma de log search consegue agregar.
