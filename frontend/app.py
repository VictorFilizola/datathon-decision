import streamlit as st
import requests
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

# --- Settings ---
API_BASE_URL = "http://backend:8000"

# --- Path Logic ---
try:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    TRAINING_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'training_dataset.json')
    PREDICTIONS_LOG_PATH = os.path.join(BASE_DIR, 'predictions.log')
except NameError:
    BASE_DIR = os.path.abspath('.')
    TRAINING_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'training_dataset.json')
    PREDICTIONS_LOG_PATH = os.path.join(BASE_DIR, 'predictions.log')

# --- Dashboard panel ---
@st.cache_data
def load_monitoring_data():
    """Carrega os dados de treinamento e os logs de predição para o painel."""
    try:
        training_df = pd.read_json(TRAINING_DATA_PATH)
    except FileNotFoundError:
        training_df = pd.DataFrame()

    try:
        log_lines = []
        with open(PREDICTIONS_LOG_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                json_part = line.split(' - ')[-1]
                log_lines.append(json.loads(json_part))
        predictions_df = pd.DataFrame(log_lines)
    except (FileNotFoundError, json.JSONDecodeError, IndexError):
        predictions_df = pd.DataFrame()
        
    return training_df, predictions_df

# --- Main UI---
st.set_page_config(layout="wide")

# --- Sidebar ---
st.sidebar.title("Navegação")
page = st.sidebar.selectbox("Escolha uma página:", ["Aplicação Principal", "Painel de Monitoramento"])

# --- Mainpage rendering ---
if page == "Aplicação Principal":
    st.title("Interface MVP do Modelo de Recrutamento")
    st.markdown("Uma interface simples para interagir com a API de correspondência de recrutamento.")

    st.sidebar.header("Ações do Modelo")
    if st.sidebar.button("Treinar Novo Modelo"):
        with st.spinner("Treinamento em andamento... Isso pode levar alguns minutos."):
            try:
                response = requests.post(f"{API_BASE_URL}/train")
                if response.status_code == 201:
                    st.sidebar.success(f"Novo modelo treinado com sucesso! Arquivo: {response.json().get('new_model_file')}")
                else:
                    st.sidebar.error(f"O treinamento falhou: {response.text}")
            except requests.exceptions.RequestException as e:
                st.sidebar.error(f"Erro de conexão com a API: {e}")

    st.header("1. Avaliar Desempenho do Modelo")
    col1, col2 = st.columns(2)

    with col1:
        try:
            models_response = requests.get(f"{API_BASE_URL}/models")
            if models_response.status_code == 200:
                available_models = models_response.json().get("models", [])
                if available_models:
                    selected_model = st.selectbox("Escolha um modelo para avaliar:", available_models)
                else:
                    st.warning("Nenhum modelo encontrado no diretório /models.")
                    selected_model = None
            else:
                st.warning("Não foi possível buscar a lista de modelos. A API está em execução?")
                selected_model = None
        except requests.exceptions.RequestException:
            st.error("Erro de conexão com a API. Por favor, garanta que o backend está em execução.")
            selected_model = None

    with col2:
        st.write("") 
        st.write("") 
        if selected_model and st.button("Avaliar Modelo Selecionado"):
            with st.spinner(f"Avaliando {selected_model}..."):
                eval_response = requests.get(f"{API_BASE_URL}/evaluate/{selected_model}")
                if eval_response.status_code == 200:
                    metrics = eval_response.json()
                    st.subheader(f"Desempenho para `{metrics['model_filename']}`")
                    st.metric("Acurácia", metrics['accuracy'])
                    st.text("Relatório de Classificação:")
                    report_df = pd.DataFrame(metrics['classification_report']).transpose()
                    st.dataframe(report_df)
                else:
                    st.error(f"A avaliação falhou: {eval_response.text}")

    st.divider()

    st.header("2. Encontrar Melhores Vagas para um Candidato")
    st.subheader("Opção 1: Cole o JSON bruto do candidato aqui")
    default_applicant_json = {
        "35000-exemplo": {
            "infos_basicas": { "codigo_profissional": "35000-exemplo" },
            "cv_pt": "Desenvolvedor de software com experiência em Python e SQL para análise de dados. Conhecimento em bibliotecas como Pandas e Scikit-learn. Inglês intermediário para leitura de documentação técnica."
        }
    }
    applicant_json_str = st.text_area("Dados do Candidato", json.dumps(default_applicant_json, indent=4, ensure_ascii=False), height=200)

    st.subheader("Opção 2: Ou faça o upload de um arquivo .json")
    uploaded_file = st.file_uploader("Carregar arquivo .json", type=["json"])

    if st.button("Encontrar Vagas para o Candidato"):
        input_json_str = ""
        if uploaded_file is not None:
            try:
                input_json_str = uploaded_file.getvalue().decode("utf-8")
                st.info("Usando o arquivo enviado para a predição.")
            except Exception as e:
                st.error(f"Erro ao ler o arquivo: {e}")
                input_json_str = ""
        else:
            input_json_str = applicant_json_str

        if not input_json_str or input_json_str == "{}":
            st.error("Por favor, cole os dados JSON do candidato ou faça o upload de um arquivo.")
        else:
            try:
                applicant_data_full = json.loads(input_json_str)
                applicant_data_to_send = next(iter(applicant_data_full.values()))
                
                with st.spinner("Encontrando as melhores correspondências..."):
                    predict_response = requests.post(f"{API_BASE_URL}/predict/latest", json=applicant_data_to_send)

                    if predict_response.status_code == 200:
                        results = predict_response.json()
                        st.success(f"Predição completa usando o modelo: `{results['model_used']}`")
                        
                        st.subheader("Características do Candidato (Extraídas pelo Gemini)")
                        extracted_features = results.get('applicant_extracted_features', {})
                        cols = st.columns(3)
                        cols[0].metric("Nível de Experiência", extracted_features.get('experience_level', 'N/A').capitalize())
                        cols[1].info(f"**Habilidades:** {', '.join(extracted_features.get('technical_skills', [])) or 'Nenhuma encontrada'}")
                        cols[2].info(f"**Idiomas:** {json.dumps(extracted_features.get('languages', {})) or 'Nenhum encontrado'}")
                        
                        st.subheader("Top 5 Vagas Correspondentes")
                        
                        for match in results.get('top_matches', []):
                            probability = match.get('match_probability', 0)
                            vaga_details = match.get('vaga_details', {})
                            with st.expander(f"**{vaga_details.get('title', 'N/A')}** - Probabilidade de Match: {probability:.2%}"):
                                st.markdown(f"**ID da Vaga:** `{match.get('vaga_id')}`")
                                st.markdown(f"**Cliente:** {vaga_details.get('client', 'N/A')}")
                                st.markdown(f"**Tipo de Contrato:** {vaga_details.get('contract_type', 'N/A')}")
                                st.markdown(f"**Pontuação de Habilidades:** {match.get('skill_match_score', 0):.2f}")
                                st.markdown("---")
                                st.markdown("**Principais Atividades:**")
                                st.info(vaga_details.get('main_activities', 'Não especificado.'))

                    else:
                        st.error(f"A predição falhou: {predict_response.text}")

            except (json.JSONDecodeError, StopIteration):
                st.error("Formato de JSON inválido. Por favor, verifique os dados. O formato esperado é {'ID_CANDIDATO': { ...dados... }}")
            except requests.exceptions.RequestException as e:
                st.error(f"Erro de conexão com a API: {e}")

    st.divider()

    st.header("3. Navegar pelo Conjunto de Dados de Treinamento")
    if st.checkbox("Carregar e Mostrar Dados de Treinamento"):
        try:
            st.info(f"Tentando carregar dados de: {TRAINING_DATA_PATH}")
            df_training = pd.read_json(TRAINING_DATA_PATH)
            st.dataframe(df_training)
        except FileNotFoundError:
            st.error(f"Não foi possível encontrar o conjunto de dados de treinamento em {TRAINING_DATA_PATH}. Por favor, certifique-se de que o arquivo existe.")
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")

# --- Dashboard panel rendering ---
elif page == "Painel de Monitoramento":
    st.title("Painel de Monitoramento de Drift do Modelo")
    
    training_df, predictions_df = load_monitoring_data()

    if training_df.empty:
        st.error(f"Arquivo de treinamento não encontrado. Verifique o caminho: {TRAINING_DATA_PATH}")
    if predictions_df.empty:
        st.warning(f"Nenhum log de predição encontrado. Execute algumas predições na 'Aplicação Principal' primeiro.")
    
    st.info(f"Carregados {len(training_df)} registros de treinamento e {len(predictions_df)} registros de predição.")

    if not training_df.empty and not predictions_df.empty:
        features_to_monitor = ['skill_match_score', 'level_match_score', 'match_probability']
        
        st.header("Comparação da Distribuição de Dados")
        
        for feature in features_to_monitor:
            if feature not in predictions_df.columns or feature not in training_df.columns:
                continue
                
            st.subheader(f"Análise de Drift para: `{feature}`")
            
            fig, ax = plt.subplots()
            
            training_df[feature].hist(ax=ax, bins=20, alpha=0.5, label='Dados de Treinamento', density=True)
            predictions_df[feature].hist(ax=ax, bins=20, alpha=0.5, label='Dados de Predição (Produção)', density=True)
            
            ax.legend()
            ax.set_title(f"Distribuição de '{feature}'")
            ax.set_xlabel("Valor")
            ax.set_ylabel("Densidade")
            
            st.pyplot(fig)
            
            mean_training = training_df[feature].mean()
            mean_prediction = predictions_df[feature].mean()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Média nos Dados de Treinamento", value=f"{mean_training:.2f}")
            with col2:
                st.metric(label="Média nos Dados de Predição", value=f"{mean_prediction:.2f}", delta=f"{mean_prediction - mean_training:.2f} (Diferença)")
            
            st.markdown("---")