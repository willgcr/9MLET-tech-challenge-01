# Arquitetura de deploy

## Decisão: inferência real-time via API REST

Escolhemos **deploy em tempo real** (sob demanda), não em batch.

### Real-time vs Batch — análise

| Critério | Batch (job diário/semanal) | Real-time (API) ✓ |
|---|---|---|
| Latência | minutos a horas | <100 ms |
| Frescor de dados | até a data do último job | momento da chamada |
| Caso de uso | scoring massivo, dashboards | priorização ad hoc, integrações |
| Custo | menor (rodada periódica) | maior (servidor sempre on) |
| Complexidade ops | scheduler + storage | servidor + observabilidade |
| Retraining | igual nos dois modelos | igual |

**Por que real-time** para esse caso:
- O time de retenção precisa **agir imediatamente** quando um cliente liga reclamando ou quando há mudança de plano.
- Integrações futuras com CRM/atendimento exigem chamada síncrona.
- Latência de inferência da MLP é trivial (~25 ms p95 em CPU local), então o custo extra do servidor é modesto.
- Para análises agregadas (dashboards), pode-se chamar a API em batch (`POST /predict/batch`) ou exportar a predição cacheada — um único endpoint serve os dois cenários.

## Topologia

```
┌──────────────────┐
│ Cliente          │   (CRM, frontend interno, BI, integração)
└─────────┬────────┘
          │ HTTPS POST /predict
          ▼
┌──────────────────┐
│ Reverse Proxy    │   nginx / Traefik / cloud LB
│ • TLS termination│
│ • rate limit     │
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│ FastAPI app      │
│ (uvicorn ASGI)   │
│ • Pydantic valid │
│ • LatencyMW logs │
│ • lifespan loads │
│   model artifacts│
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│ models/          │
│ ├ preprocessor   │   sklearn pipeline (joblib)
│ ├ mlp.pt         │   torch state dict
│ └ metadata.json  │   threshold, version, op points
└──────────────────┘
```

## Stack de execução

- **Linguagem**: Python 3.10.
- **Framework**: FastAPI 0.115 + Uvicorn (ASGI).
- **Serialização**: `joblib` (sklearn pipeline), `torch.save` (state dict), `json` (metadata).
- **Container**: Docker (multi-stage build mínimo: stage 1 instala deps via `uv`, stage 2 copia só o pacote + venv + artefatos).
- **Resource sizing inicial**: 1 vCPU, 512 MB RAM, 1 réplica. Escalar horizontalmente conforme demanda — o modelo é stateless.
- **Health check**: `GET /health` retorna `{"status":"ok","model_loaded":true,"threshold":0.4399,"model_version":"ChurnMLP"}`. Reverse proxy usa para drenar instâncias com falha.

## Reprodutibilidade

- **`uv.lock`** trava todas as dependências transitivas — `uv sync` produz o mesmo ambiente em qualquer máquina.
- **`models/`** carrega o modelo já treinado; novas builds podem reusar artefatos antigos ou rodar `make train` para regenerar.
- **`make train`** reproduz o treino do zero com seed fixa (`RANDOM_SEED=42`).

## Plano de deploy

| Etapa | Comando / mecanismo | Status |
|---|---|---|
| 1. Local | `make serve` | implementado e testado |
| 2. Container | `docker build -t churn-api . && docker run -p 8000:8000 churn-api` | Dockerfile vazio nesse ponto, fechar antes de subir |
| 3. VPS dedicado | `docker compose up -d` atrás de um reverse proxy (Traefik / nginx) com TLS via Let's Encrypt | proposta — não implementado neste escopo |
| 4. Cloud (futuro) | AWS ECS/Fargate, GCP Cloud Run, Azure Container Apps | a imagem é o deliverable; a plataforma é commodity |

## Failure modes considerados

| Falha | Detecção | Mitigação |
|---|---|---|
| Modelo não carrega no startup | lifespan crasha → container não fica pronto | orchestrator reinicia; `/health` retorna 503 até carregar |
| Out-of-memory | container morre | sizing + replicação; alerta de RSS |
| Categoria desconhecida na request | Pydantic rejeita com 422 | mensagem com campo inválido na resposta |
| `TotalCharges` em formato inesperado | `FeatureEngineer` coage para NaN, imputer trata | predição segue, sem erro |
| Spike de tráfego | latência sobe | autoscaling horizontal, timeout no proxy |
| Modelo defasado (drift) | métricas de monitoramento (ver `monitoring.md`) | retreinar (`make train`) e republicar artefato |

## Segurança

- **TLS** terminado no reverse proxy.
- **Rate limit** no proxy para evitar abuso.
- **Sem autenticação no endpoint de predição neste escopo** — em produção, adicionar API key ou JWT (opção descrita na disciplina de APIs do curso). FastAPI suporta nativamente via `Depends(...)`.
- **Sem PII no payload** — apenas features tabulares de produto/uso.
- **Logs JSON** evitam log de payloads completos por padrão (só campos do `extra`).
