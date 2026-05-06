# Telco Churn — Tech Challenge Fase 01

Pipeline end-to-end de previsão de churn para uma operadora de telecomunicações: rede neural (MLP em PyTorch) comparada com baselines em Scikit-Learn, experimentos rastreados em MLflow e modelo servido via FastAPI.

> Entrega da **Fase 01 — Produtização de Modelos** do curso de Pós-Graduação em Machine Learning Engineering (FIAP / POS TECH — 9MLET).

> 🌐 **API em produção:** https://9mlet.willgcr.me — documentação interativa em [`/docs`](https://9mlet.willgcr.me/docs) (Swagger) e [`/redoc`](https://9mlet.willgcr.me/redoc).

## Sumário

- [Visão geral](#visão-geral)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Setup](#setup)
- [Execução](#execução)
- [Testes e qualidade de código](#testes-e-qualidade-de-código)
- [Deploy](#deploy)
- [Documentação adicional](#documentação-adicional)

## Visão geral

O dataset utilizado é o [Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), publicado no Kaggle (slug `blastchar/telco-customer-churn`). A tarefa é classificação binária: prever se um cliente irá cancelar o serviço. Volumetria, distribuição e qualidade dos dados são caracterizadas na EDA da Etapa 1 e documentadas em `docs/`.

O pipeline cobre:
1. EDA + ML Canvas (definição de stakeholders, métricas de negócio e SLOs).
2. Baselines (`DummyClassifier`, `LogisticRegression`) com validação cruzada estratificada e tracking no MLflow.
3. Rede neural MLP em PyTorch com early stopping, comparada a baselines lineares e ensembles.
4. Análise de trade-off de custo (falso positivo vs. falso negativo).
5. API FastAPI com endpoints `/health` e `/predict`, validação Pydantic e logging estruturado.
6. Empacotamento, testes (smoke + schema + API), Makefile, lint com `ruff`.

## Estrutura do repositório

```
.
├── src/churn/        # código-fonte do pacote
│   ├── dataset/      # download, schema, pré-processamento
│   ├── modeling/     # MLP, baselines, treino, avaliação
│   ├── tracking/     # helpers do MLflow
│   ├── api/          # aplicação FastAPI
│   └── cli/          # entry points (churn-train, churn-serve)
├── data/             # datasets (raw/processed/interim) - não versionados
├── models/           # artefatos treinados - não versionados
├── notebooks/        # EDA e demo end-to-end
├── tests/            # pytest (smoke, schema com pandera, API)
├── docs/             # Model Card, ML Canvas, deploy, monitoramento
├── pyproject.toml    # single source of truth (deps, lint, testes)
├── uv.lock           # pinning bit-a-bit das dependências
├── Makefile          # atalhos para tarefas comuns
└── Dockerfile        # imagem de produção da API
```

## Setup

### Pré-requisitos

- Python 3.10 (`pyproject.toml` exige `>=3.10,<3.12`).
- [`uv`](https://docs.astral.sh/uv/) instalado (`brew install uv` no macOS).

### Instalação

```bash
git clone https://github.com/willgcr/9MLET-tech-challenge-01.git
cd 9MLET-tech-challenge-01
make install        # equivalente a: uv sync --extra dev
```

O comando cria a venv em `.venv/`, instala todas as dependências travadas pelo `uv.lock` e instala o pacote em modo editável.

> Para usuários sem `uv`: `pip install -r requirements.txt` continua funcionando (o arquivo é apenas um ponteiro para `pyproject.toml`).

## Execução

| Tarefa | Comando |
|---|---|
| Treinar todos os modelos e logar no MLflow | `make train` |
| Subir a API de inferência (porta 8000) | `make serve` |
| Abrir a UI do MLflow (porta 5000) | `make mlflow-ui` |
| Construir a imagem Docker | `make docker` |
| Subir o container localmente | `make docker-up` |
| Derrubar o container | `make docker-down` |

A API expõe documentação interativa em `http://localhost:8000/docs` (Swagger) e `/redoc`. A versão em produção está hospedada em <https://9mlet.willgcr.me/docs>.

### Deploy

CI/CD via GitHub Actions (`.github/workflows/ci.yml`). Em todo push para `master`, a pipeline executa lint + testes; quando passa, faz SSH em um servidor privado e roda `git pull` + `docker compose up -d --build`. O servidor termina TLS com certificado Let's Encrypt e expõe a API em <https://9mlet.willgcr.me>.

## Testes e qualidade de código

```bash
make lint     # ruff check
make format   # ruff format + ruff --fix
make test     # pytest com cobertura
```

A suíte de testes cobre, no mínimo:
- **Smoke test** — pacote importa e versão é exposta.
- **Schema test** (`pandera`) — validação do DataFrame de entrada.
- **API test** — endpoint `/health` responde e `/predict` retorna formato esperado.

## Documentação adicional

- [`docs/model_card.md`](docs/model_card.md) — Model Card com performance, limitações e vieses.
- [`docs/ml_canvas.md`](docs/ml_canvas.md) — ML Canvas (stakeholders, métricas, SLOs).
- [`docs/deploy.md`](docs/deploy.md) — arquitetura de deploy escolhida e justificativa.
- [`docs/monitoring.md`](docs/monitoring.md) — plano de monitoramento, métricas e playbook de resposta.

## Licença

MIT.
