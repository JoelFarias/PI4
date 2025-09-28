import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError # Adicionado para tratamento de erro SQL
import plotly.express as px
import plotly.graph_objects as go
import logging

# Adicionando bibliotecas de Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
import numpy as np

# Configuração de Log (Boa prática de engenharia de dados)
logging.basicConfig(level=logging.INFO)

# --- 1. Configurações e Variáveis ---

DB_CONFIG = {
    'host': 'bigdata.dataiesb.com',
    'database': 'iesb',
    'port': 5432,
    'user': 'data_iesb',
    'password': 'iesb'
}

# Definições de Mapeamento Categórico (Melhoria de Legibilidade)
Q005_MAP = {
    'A': 'Nenhuma Renda', 'B': 'Até R$ 1.320,00', 'C': 'De R$ 1.320,01 a R$ 1.980,00', 
    'D': 'De R$ 1.980,01 a R$ 2.640,00', 'E': 'De R$ 2.640,01 a R$ 3.300,00', 
    'F': 'De R$ 3.300,01 a R$ 3.960,00', 'G': 'De R$ 3.960,01 a R$ 5.280,00', 
    'H': 'De R$ 5.280,01 a R$ 6.600,00', 'I': 'De R$ 6.600,01 a R$ 7.920,00', 
    'J': 'De R$ 7.920,01 a R$ 9.240,00', 'K': 'De R$ 9.240,01 a R$ 10.560,00', 
    'L': 'De R$ 10.560,01 a R$ 11.880,00', 'M': 'De R$ 11.880,01 a R$ 13.200,00', 
    'N': 'De R$ 13.200,01 a R$ 15.840,00', 'O': 'De R$ 15.840,01 a R$ 19.800,00', 
    'P': 'Mais de R$ 19.800,00', 'Q': 'Não Declarado' 
}

# Mapeamento de Escolaridade
escolaridade_map = {
    'A': 'A - Nenhuma/Incompleto', 'B': 'B - Fund. Completo', 
    'C': 'C - Médio Incompleto', 'D': 'D - Médio Completo', 
    'E': 'E - Superior Incompleto', 'F': 'F - Superior Completo', 
    'G': 'G - Pós-Graduação', 'H': 'H - Não Sabe'
}

# --- 2. Funções de Carga e Mapeamento de Dados ---

@st.cache_data(show_spinner="Conectando e carregando amostra do ENEM 2024...")
def load_data(sample_size=2000):
    """
    Carrega uma amostra aleatória dos dados do ENEM 2024 do PostgreSQL,
    realizando um JOIN entre participantes (socioeconômico) e resultados (notas).
    """
    try:
        connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        engine = create_engine(connection_string, pool_pre_ping=True)
        
        # SQL CORRIGIDO: Realiza INNER JOIN entre participantes (p) e resultados (r)
        # Atenção: a chave de junção 'nu_sequencial' é uma ASSUNÇÃO, se falhar, este é o ponto a ser verificado.
        query = f"""
            SELECT 
                -- Socioeconômico de participantes (p)
                p.q001 as escolaridade_pai,
                p.q002 as escolaridade_mae,
                p.q005 as faixa_renda,
                p.tp_sexo as sexo,
                p.tp_cor_raca as cor_raca,
                p.idade_calculada as idade,
                
                -- Notas e Locais de Resultados (r)
                r.sg_uf_prova as uf,
                r.regiao_nome_prova as regiao,
                r.nota_cn_ciencias_da_natureza,
                r.nota_ch_ciencias_humanas,
                r.nota_lc_linguagens_e_codigos,
                r.nota_mt_matematica,
                r.nota_redacao,
                r.nota_media_5_notas
            FROM ed_enem_2024_participantes p
            INNER JOIN ed_enem_2024_resultados r 
                ON CAST(p.nu_sequencial AS VARCHAR) = CAST(r.nu_sequencial AS VARCHAR)
            WHERE p.q001 IS NOT NULL 
              AND p.q002 IS NOT NULL 
              AND p.q005 IS NOT NULL
              AND r.nota_media_5_notas IS NOT NULL AND r.nota_media_5_notas > 0
            ORDER BY RANDOM()
            LIMIT {sample_size};
        """
        logging.info(f"Executando query com JOIN e LIMIT {sample_size}")
        df = pd.read_sql(query, engine)
        engine.dispose()
        return df
    except SQLAlchemyError as e:
        # TRATAMENTO DE ERRO MELHORADO: Exibe o erro SQL específico
        error_message = f"🚨 Erro SQL ao carregar dados. Verifique a sintaxe da QUERY ou a chave de junção ('nu_sequencial'). Detalhes: {e}"
        logging.error(error_message)
        st.error(error_message)
        return pd.DataFrame()
    except Exception as e:
        # Erro geral (conexão, pandas, etc.)
        error_message = f"❌ Erro geral ao carregar dados. Verifique a conexão. Detalhes: {e}"
        logging.error(error_message)
        st.error(error_message)
        return pd.DataFrame()

def decode_enem_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica mapeamentos para tornar variáveis categóricas legíveis e preparar para o dashboard.
    """
    df['faixa_renda_legivel'] = df['faixa_renda'].map(Q005_MAP).fillna('Desconhecido')
    df['sexo'] = df['sexo'].replace({'M': 'Masculino', 'F': 'Feminino'})
    
    df['escolaridade_pai'] = df['escolaridade_pai'].replace(escolaridade_map)
    df['escolaridade_mae'] = df['escolaridade_mae'].replace(escolaridade_map)

    return df

# --- 3. Funções de Visualização (Adicionado) ---

def create_pie_chart(df, column, title):
    """
    Cria um gráfico de pizza para a coluna categórica especificada.
    """
    value_counts = df[column].value_counts(dropna=False)
    fig = px.pie(
        names=value_counts.index,
        values=value_counts.values,
        title=title,
        hole=0.3
    )
    fig.update_traces(textinfo='percent+label')
    return fig

def create_income_bar_chart(df):
 
    renda_legivel = df['faixa_renda_legivel'].value_counts().sort_index()
    fig = px.bar(
        x=renda_legivel.index,
        y=renda_legivel.values,
        labels={'x': 'Faixa de Renda', 'y': 'Quantidade'},
        title='Distribuição por Faixa de Renda',
        text=renda_legivel.values
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_notes_box_plot(df):
    """
    Cria um boxplot das notas do ENEM por disciplina.
    """
    notas_cols = [
        "nota_cn_ciencias_da_natureza",
        "nota_ch_ciencias_humanas",
        "nota_lc_linguagens_e_codigos",
        "nota_mt_matematica",
        "nota_redacao"
    ]
    notas_legendas = [
        "Ciências da Natureza",
        "Ciências Humanas",
        "Linguagens e Códigos",
        "Matemática",
        "Redação"
    ]
    data = []
    for col, legenda in zip(notas_cols, notas_legendas):
        if col in df.columns:
            data.append(go.Box(
                y=df[col].dropna(),
                name=legenda,
                boxmean=True
            ))
    fig = go.Figure(data=data)
    fig.update_layout(
        title="Distribuição das Notas por Disciplina",
        yaxis_title="Nota",
        boxmode="group"
    )
    return fig

def create_income_vs_math_box_plot(df):
    """
    Cria um boxplot da nota de matemática por faixa de renda legível.
    """
    if "faixa_renda_legivel" not in df.columns or "nota_mt_matematica" not in df.columns:
        return go.Figure()
    fig = px.box(
        df,
        x="faixa_renda_legivel",
        y="nota_mt_matematica",
        title="Nota de Matemática por Faixa de Renda",
        labels={"faixa_renda_legivel": "Faixa de Renda", "nota_mt_matematica": "Nota de Matemática"},
        points="outliers"
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

# --- 3. Funções de Análise Preditiva (Novo) ---

def perform_predictive_analysis(df: pd.DataFrame):
    """
    Realiza a Análise Preditiva (Regressão Linear) para prever a nota média
    com base nas variáveis socioeconômicas e retorna os coeficientes.
    """
    target = 'nota_media_5_notas'
    features = ['faixa_renda', 'escolaridade_pai', 'escolaridade_mae'] 
    
    df_clean = df.dropna(subset=features + [target]).copy()

    if len(df_clean) < 100:
        return 0, pd.DataFrame(columns=['Variável', 'Coeficiente', 'Impacto'])

    X = df_clean[features]
    y = df_clean[target]

    # Pré-processamento: One-Hot Encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features)
        ],
        remainder='passthrough'
    )

    # Split e Treinamento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_processed, y_train)

    # Avaliação
    y_pred = model.predict(X_test_processed)
    r2 = r2_score(y_test, y_pred)
    
    # Extrair Coeficientes para Relatório Analítico
    feature_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(features)
    
    coefficients_df = pd.DataFrame({
        'Variável': feature_names,
        'Coeficiente': model.coef_
    })
    
    # Adicionar o mapeamento legível de volta
    def get_legible_name(var_code):
        parts = var_code.split('_')
        feature = parts[0]
        code = parts[-1]
        
        if feature == 'faixa':
            return Q005_MAP.get(code, f"Renda {code}")
        elif feature == 'escolaridade':
            # Determina se é pai ou mãe
            parent_tag = 'Mãe' if 'mae' in var_code else 'Pai'
            return f"{parent_tag}: {escolaridade_map.get(code, code)}"
        return var_code

    coefficients_df['Variável'] = coefficients_df['Variável'].apply(get_legible_name)
    coefficients_df['Impacto'] = np.where(coefficients_df['Coeficiente'] > 0, "POSITIVO", "NEGATIVO")
    
    intercept_df = pd.DataFrame({'Variável': ['Base (Intercepto)'], 'Coeficiente': [model.intercept_], 'Impacto': ['BASE']})
    coefficients_df = pd.concat([intercept_df, coefficients_df], ignore_index=True)

    return r2, coefficients_df.sort_values(by='Coeficiente', ascending=False)

# --- 4. Layout do Streamlit ---

# Configuração da página e título
st.set_page_config(page_title="ENEM 2024 - Dashboard Socioeconômico", layout="wide")
st.title("📊 ENEM 2024 - Dashboard Socioeconômico")
st.markdown("---")

# Carga e Mapeamento dos Dados
df_raw = load_data(sample_size=2000)
if df_raw.empty:
    st.stop()

# Aplica o Mapeamento
df = decode_enem_categories(df_raw.copy())
st.success(f"✅ Dados carregados e processados: {len(df):,} registros na amostra!")

# --- 5. SideBar de Filtros ---

st.sidebar.header("Filtros do Dashboard")

# Adicione seleção de tamanho da amostra na barra lateral
sample_size = st.sidebar.number_input(
    "Tamanho da amostra (máx. 10.000)", min_value=500, max_value=10000, value=2000, step=500,
    help="Reduza para carregar mais rápido. Aumente para mais precisão nas análises."
)

regioes = ["Todas"] + sorted(df["regiao"].dropna().unique())
ufs = ["Todas"] + sorted(df["uf"].dropna().unique())
generos = ["Todos"] + sorted(df["sexo"].dropna().unique())
rendas_unicas = sorted(df["faixa_renda"].dropna().unique(), key=lambda x: list(Q005_MAP.keys()).index(x) if x in Q005_MAP else 99)
rendas_legiveis = [Q005_MAP.get(r, r) for r in rendas_unicas]
rendas_map_rev = {v: k for k, v in Q005_MAP.items()}

regiao_sel = st.sidebar.selectbox("Região", regioes)
uf_sel = st.sidebar.selectbox("UF", ufs)
genero_sel = st.sidebar.selectbox("Sexo", generos)
renda_sel_legivel = st.sidebar.selectbox("Faixa de Renda", ["Todas"] + rendas_legiveis)

# --- 6. Aplicação de Filtros ---

df_filtrado = df.copy()

if regiao_sel != "Todas":
    df_filtrado = df_filtrado[df_filtrado["regiao"] == regiao_sel]
if uf_sel != "Todas":
    df_filtrado = df_filtrado[df_filtrado["uf"] == uf_sel]
if genero_sel != "Todos":
    df_filtrado = df_filtrado[df_filtrado["sexo"] == genero_sel]
if renda_sel_legivel != "Todas":
    renda_sel_codigo = rendas_map_rev.get(renda_sel_legivel)
    if renda_sel_codigo:
        df_filtrado = df_filtrado[df_filtrado["faixa_renda"] == renda_sel_codigo]

st.info(f"Filtros aplicados. Registros para análise: **{len(df_filtrado):,}**")
st.markdown("---")

if len(df_filtrado) == 0:
    st.warning("⚠️ Nenhum registro encontrado com os filtros selecionados. Tente expandir sua seleção.")
    st.stop()

# --- 7. Exibição dos Gráficos (Análises Exploratórias - Passo 3 e 4) ---

tab1, tab2 = st.tabs(["Análise Exploratória", "Análise Preditiva e Relatório"])

with tab1:
    st.subheader("Análise Socioeconômica e Notas do ENEM")
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = create_pie_chart(df_filtrado, "escolaridade_pai", "Escolaridade do Pai")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = create_pie_chart(df_filtrado, "escolaridade_mae", "Escolaridade da Mãe")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        fig3 = create_income_bar_chart(df_filtrado)
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        fig4 = create_notes_box_plot(df_filtrado)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    col5, col6 = st.columns([3, 1])
    with col5:
        fig5 = create_income_vs_math_box_plot(df_filtrado)
        st.plotly_chart(fig5, use_container_width=True)
    with col6:
        fig6 = create_pie_chart(df_filtrado, "sexo", "Distribuição por Sexo")
        st.plotly_chart(fig6, use_container_width=True)
        
    with st.expander("🔍 Detalhamento das Estatísticas"):
        st.subheader("Estatísticas Descritivas das Notas")
        st.dataframe(df_filtrado[[
            "nota_cn_ciencias_da_natureza", "nota_ch_ciencias_humanas",
            "nota_lc_linguagens_e_codigos", "nota_mt_matematica",
            "nota_redacao", "nota_media_5_notas"
        ]].describe().style.format(precision=2), use_container_width=True)

        st.subheader("Distribuição de Idades")
        fig_idade = px.histogram(df_filtrado, x="idade", nbins=30, title="Distribuição de Idades dos Participantes")
        st.plotly_chart(fig_idade, use_container_width=True)
        
        st.subheader("Distribuição por Cor/Raça")
        fig_cor = create_pie_chart(df_filtrado, "cor_raca", "Distribuição por Cor/Raça")
        st.plotly_chart(fig_cor, use_container_width=True)

# --- 8. Análise Preditiva e Relatório Interativo (Passo 5 e 6) ---

with tab2:
    st.subheader("🔬 Análise Preditiva: Impacto Socioeconômico na Nota Média")
    st.markdown("""
        O modelo de **Regressão Linear Múltipla** foi treinado para prever a **Nota Média (Target)** usando a **Renda** e a **Escolaridade dos Pais (Preditores)**.
        Os coeficientes mostram o impacto médio em pontos na nota para cada categoria.
    """)

    r2, coefficients_df = perform_predictive_analysis(df_filtrado)

    if r2 == 0:
        st.warning("⚠️ Dados insuficientes (menos de 100 registros) para treinar o modelo preditivo.")
    else:
        col_r2, col_rmse = st.columns(2)
        with col_r2:
            st.metric(
                label="Coeficiente de Determinação ($R^2$)", 
                value=f"{r2:.4f}",
                help="Proporção da variância total das notas que é explicada pelas variáveis socioeconômicas no modelo."
            )
        with col_rmse:
             st.metric(
                label="Tamanho da Amostra para Predição", 
                value=f"{len(df_filtrado):,}",
                help="O modelo foi treinado com o subset de dados atual."
            )

        st.markdown("### Relatório Analítico: Coeficientes do Modelo")
        st.markdown("""
            **Interpretação:** Cada coeficiente representa o **impacto médio em pontos na Nota Média do ENEM**. 
            Um valor positivo significa um aumento na nota em relação ao grupo de referência implícito pelo OHE.
        """)

        st.dataframe(
            coefficients_df.style.format({
                "Coeficiente": "{:+.2f}"
            }), 
            use_container_width=True,
            height=600
        )

st.markdown("---")
with st.expander("📄 Ver Dados Brutos Filtrados"):
    st.dataframe(df_filtrado, use_container_width=True)

st.caption("Dashboard ENEM 2024 - Desenvolvido em Python/Streamlit. Dados: PostgreSQL.")
st.caption("Dashboard ENEM 2024 - Desenvolvido em Python/Streamlit. Dados: PostgreSQL.")
