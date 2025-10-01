# Projeto Datathon: Sistema de Recrutamento Inteligente

## Visão Geral

Este projeto é uma solução de Machine Learning de ponta a ponta desenvolvida como parte do Datathon PÓS TECH. O objetivo é otimizar o processo de recrutamento e seleção da empresa "Decision", utilizando Inteligência Artificial para realizar o "match" inteligente entre candidatos e vagas.

A aplicação consiste em:
1.  Um **pipeline de processamento de dados** que extrai características estruturadas de currículos e descrições de vagas.
2.  Um **modelo de Machine Learning** (Random Forest) treinado para prever a probabilidade de um candidato ser contratado para uma determinada vaga.
3.  Uma **API RESTful** construída com FastAPI para servir o modelo e gerenciar seu ciclo de vida (treinamento, avaliação).
4.  Uma **interface web interativa** construída com Streamlit que consome a API, permitindo que os recrutadores testem o sistema.
5.  Um **painel de monitoramento de drift** para acompanhar a performance do modelo ao longo do tempo.
6.  Toda a aplicação é **containerizada com Docker**, garantindo portabilidade e facilidade de implantação.

## Stack Tecnológica

* **Linguagem:** Python 3.11
* **Backend API:** FastAPI
* **Frontend & Dashboard:** Streamlit
* **Machine Learning:** Scikit-learn, Pandas
* **Containerização:** Docker, Docker Compose
* **Servidor:** Uvicorn

## Estrutura do Projeto

```
.
├── backend/
│   └── main.py             # Lógica da API FastAPI (endpoints /train, /predict, etc.)
├── data/
│   ├── processed/          # Datasets intermediários e finais (ex: training_dataset.json)
│   └── raw/                # Dados brutos e imutáveis (applicants.json, etc.)
├── frontend/
│   └── app.py              # Interface do usuário e painel de monitoramento (Streamlit)
├── models/
│   └── ...                 # Modelos de ML treinados e versionados (arquivos .joblib)
├── src/
│   └── ml/
│       ├── build_dataset.py          # Script para agregar dados brutos
│       ├── feature_extractor.py      # Lógica de extração de features (simulação de LLM)
│       ├── create_training_data.py   # Script para criar o dataset de treinamento
│       └── train.py                  # Script para treinar e avaliar o modelo de ML
├── tests/
│   ├── test_api.py         # Testes automatizados para a API
│   └── test_ml.py          # Testes unitários para a lógica de ML
├── .dockerignore           # Arquivos a serem ignorados pelo Docker
├── .env                    # Arquivo para variáveis de ambiente (ex: API keys)
├── .gitignore              # Arquivos a serem ignorados pelo Git
├── Dockerfile              # Blueprint para construir a imagem Docker da aplicação
├── docker-compose.yml      # Orquestrador para rodar os serviços de backend e frontend
├── predictions.log         # Arquivo de log para monitoramento de drift
├── requirements.txt        # Dependências Python do projeto
└── start_app.bat           # Script para iniciar a aplicação localmente no Windows
```

## Como Executar

Existem duas maneiras de executar este projeto: localmente ou via Docker (recomendado).

### Método 1: Execução Local (Windows)

**Pré-requisitos:** Python 3.11+ instalado.

1.  **Instale as Dependências:** Abra um terminal na raiz do projeto e execute:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Inicie a Aplicação:** Dê um duplo clique no arquivo `start_app.bat`.
    * Isso iniciará o servidor da API e a interface do frontend em janelas de terminal separadas e abrirá a aplicação no seu navegador.
3.  **Acesse a Interface:** Se o navegador não abrir automaticamente, acesse: **`http://localhost:8501`**

### Método 2: Execução com Docker (Recomendado)

**Pré-requisitos:** Docker Desktop instalado e em execução.

1.  **Abra um Terminal:** Navegue até a raiz do projeto.
2.  **Construa e Inicie os Containers:** Execute o seguinte comando:
    ```bash
    docker-compose up --build
    ```
    * Este comando irá construir a imagem Docker e iniciar os containers do backend e do frontend. O primeiro build pode levar alguns minutos.
3.  **Acesse a Interface:** Abra seu navegador e acesse: **`http://localhost:8501`**

## Pipeline de Machine Learning

O processo de transformação dos dados brutos em um modelo preditivo é dividido em quatro etapas principais, orquestradas pelos scripts no diretório `src/ml/`:

1.  **Agregação de Dados (`build_dataset.py`):** Inicialmente, um script seleciona uma amostra de `prospects` e agrega as informações completas das vagas (`vagas.json`) e dos candidatos (`applicants.json`) em um único arquivo (`prospects_aggregated.json`), que serve como base para o processamento.

2.  **Extração de Features (`feature_extractor.py`):** Este script simula a funcionalidade de um LLM (como o Gemini). Ele lê os textos não estruturados (CVs e descrições de vagas) e extrai informações valiosas e estruturadas, como habilidades técnicas, nível de experiência e idiomas, salvando-as nos arquivos `applicants_enhanced.json` e `vacancies_enhanced.json`.

3.  **Criação do Dataset de Treinamento (`create_training_data.py`):** Utilizando os dados estruturados e o mapa de candidaturas, este script monta o dataset final. Ele não apenas combina os dados, mas também realiza a **engenharia de features**, criando métricas comparativas como `skill_match_score` (percentual de habilidades compatíveis) e `level_match_score` (compatibilidade de senioridade). A variável alvo `hired` é criada aqui.

4.  **Treinamento e Versionamento (`train.py`):** O script final carrega o dataset de treinamento, divide-o em conjuntos de treino e teste, treina um modelo `RandomForestClassifier` e avalia sua performance. O modelo treinado é salvo na pasta `models/` com um timestamp no nome para versionamento.

## API Endpoints

A API FastAPI fornece uma interface para interagir com o sistema de ML.

### Acesso à Documentação Interativa (Swagger UI)

A melhor maneira de explorar e testar a API é através da documentação interativa gerada automaticamente. Com a aplicação em execução, acesse:

**`http://localhost:8000/docs`**

### Rotas Disponíveis

* **`POST /train`**: Inicia o processo de retreinamento do modelo. Salva um novo arquivo `.joblib` na pasta `models/`.
* **`GET /models`**: Retorna uma lista de todos os modelos treinados e disponíveis.
* **`GET /evaluate/{model_filename}`**: Avalia um modelo específico usando o conjunto de teste e retorna suas métricas de performance (Acurácia, Precisão, Recall, etc.).
* **`POST /predict/{model_filename}`**: O principal endpoint de predição.
    * Recebe o JSON bruto de um candidato no corpo da requisição.
    * Usa o modelo especificado (ou `"latest"` para o mais recente) para calcular a probabilidade de "match" com todas as vagas disponíveis.
    * Retorna um Top 5 das vagas mais recomendadas, enriquecidas com detalhes da vaga e as features extraídas do candidato.

## Testes

O projeto inclui testes automatizados para garantir a qualidade do código. Para executá-los, abra um terminal na raiz do projeto e use os seguintes comandos:

* **Testar a lógica do pipeline de ML:**
    ```bash
    python tests/test_ml.py
    ```
* **Testar todos os endpoints da API:**
    ```bash
    python tests/test_api.py
    ```