import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import plotly.express as px
import plotly.graph_objects as go
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
import numpy as np

logging.basicConfig(level=logging.INFO)

DB_CONFIG = {
    'host': 'bigdata.dataiesb.com',
    'database': 'iesb',
    'port': 5432,
    'user': 'data_iesb',
    'password': 'iesb'
}

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

escolaridade_map = {
    'A': 'A - Nenhuma/Incompleto', 'B': 'B - Fund. Completo', 
    'C': 'C - Médio Incompleto', 'D': 'D - Médio Completo', 
    'E': 'E - Superior Incompleto', 'F': 'F - Superior Completo', 
    'G': 'G - Pós-Graduação', 'H': 'H - Não Sabe'
}

@st.cache_data(show_spinner="Conectando e carregando amostra do ENEM 2024...")
def load_data(sample_size=100000):
    try:
        connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        engine = create_engine(connection_string, pool_pre_ping=True)
        
        query = f"""
            SELECT 
                p.q001 as escolaridade_pai,
                p.q002 as escolaridade_mae,
                p.q005 as faixa_renda,
                p.tp_sexo as sexo,
                p.tp_cor_raca as cor_raca,
                p.idade_calculada as idade,
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
                ON p.nu_sequencial = r.nu_sequencial::integer 
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
        error_message = f"🚨 Erro SQL ao carregar dados. Verifique a sintaxe da QUERY ou a chave de junção ('nu_sequencial'). Detalhes: {e}"
        logging.error(error_message)
        st.error(error_message)
        return pd.DataFrame()
    except Exception as e:
        error_message = f"❌ Erro geral ao carregar dados. Verifique a conexão. Detalhes: {e}"
        logging.error(error_message)
        st.error(error_message)
        return pd.DataFrame()

def decode_enem_categories(df: pd.DataFrame) -> pd.DataFrame:
    df['faixa_renda_legivel'] = df['faixa_renda'].map(Q005_MAP).fillna('Desconhecido')
    df['sexo'] = df['sexo'].replace({'M': 'Masculino', 'F': 'Feminino'})
    df['escolaridade_pai'] = df['escolaridade_pai'].replace(escolaridade_map)
    df['escolaridade_mae'] = df['escolaridade_mae'].replace(escolaridade_map)
    return df

def create_pie_chart(df, column, title):
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "count"]
    fig = px.pie(counts, names=column, values="count", title=title, hole=0.4)
    fig.update_layout(legend_title_text=column, margin=dict(t=30, b=0, l=0, r=0))
    return fig

def create_income_bar_chart(df):
    renda_counts = df["faixa_renda_legivel"].value_counts().reset_index()
    renda_counts.columns = ["faixa_renda_legivel", "count"]
    ordered_categories = [Q005_MAP[k] for k in Q005_MAP.keys() if Q005_MAP[k] in renda_counts["faixa_renda_legivel"].tolist()]

    fig = px.bar(
        renda_counts,
        x="faixa_renda_legivel", y="count",
        title="Distribuição de Renda (Q005)",
        labels={"faixa_renda_legivel": "Faixa de Renda", "count": "Quantidade"},
        category_orders={"faixa_renda_legivel": ordered_categories}
    )
    fig.update_xaxes(tickangle=45)
    return fig

def create_notes_box_plot(df):
    notes_columns = [
        "nota_cn_ciencias_da_natureza", "nota_ch_ciencias_humanas",
        "nota_lc_linguagens_e_codigos", "nota_mt_matematica", "nota_redacao"
    ]
    df_melt = df[notes_columns].melt(var_name="Área", value_name="Nota")
    fig = px.box(
        df_melt,
        y="Nota", color="Área",
        title="Distribuição das Notas por Área de Conhecimento"
    )
    return fig

def create_income_vs_math_box_plot(df):
    ordered_categories = [Q005_MAP[k] for k in Q005_MAP.keys() if Q005_MAP[k] in df["faixa_renda_legivel"].tolist()]
    
    fig = px.box(
        df,
        x="faixa_renda_legivel",
        y="nota_mt_matematica",
        title="Impacto da Renda na Nota de Matemática (Box Plot)",
        labels={"faixa_renda_legivel": "Faixa de Renda", "nota_mt_matematica": "Nota Matemática"},
        color="faixa_renda_legivel",
        category_orders={"faixa_renda_legivel": ordered_categories}
    )
    fig.update_xaxes(tickangle=45)
    return fig

def perform_predictive_analysis(df: pd.DataFrame):
    target = 'nota_media_5_notas'
    features = ['faixa_renda', 'escolaridade_pai', 'escolaridade_mae'] 
    
    df_clean = df.dropna(subset=features + [target]).copy()

    if len(df_clean) < 100:
        return 0, pd.DataFrame(columns=['Variável', 'Coeficiente', 'Impacto'])

    X = df_clean[features]
    y = df_clean[target]

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features)
        ],
        remainder='passthrough'
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_processed, y_train)

    y_pred = model.predict(X_test_processed)
    r2 = r2_score(y_test, y_pred)
    
    feature_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(features)
    
    coefficients_df = pd.DataFrame({
        'Variável': feature_names,
        'Coeficiente': model.coef_
    })
    
    def get_legible_name(var_code):
        parts = var_code.split('_')
        feature = parts[0]
        code = parts[-1]
        
        if feature == 'faixa':
            return Q005_MAP.get(code, f"Renda {code}")
        elif feature == 'escolaridade':
            parent_tag = 'Mãe' if 'mae' in var_code else 'Pai'
            return f"{parent_tag}: {escolaridade_map.get(code, code)}"
        return var_code

    coefficients_df['Variável'] = coefficients_df['Variável'].apply(get_legible_name)
    coefficients_df['Impacto'] = np.where(coefficients_df['Coeficiente'] > 0, "POSITIVO", "NEGATIVO")
    
    intercept_df = pd.DataFrame({'Variável': ['Base (Intercepto)'], 'Coeficiente': [model.intercept_], 'Impacto': ['BASE']})
    coefficients_df = pd.concat([intercept_df, coefficients_df], ignore_index=True)

    return r2, coefficients_df.sort_values(by='Coeficiente', ascending=False)

st.set_page_config(page_title="ENEM 2024 - Dashboard Socioeconômico", layout="wide")
st.title("📊 ENEM 2024 - Dashboard Socioeconômico")
st.markdown("---")

df_raw = load_data()
if df_raw.empty:
    st.stop()

df = decode_enem_categories(df_raw.copy())
st.success(f"✅ Dados carregados e processados: {len(df):,} registros na amostra!")

st.sidebar.header("Filtros do Dashboard")

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

st.header("Análise Descritiva Rápida")
col_met1, col_met2, col_met3, col_met4 = st.columns(4)

if len(df_filtrado) > 0:
    media_geral = df_filtrado['nota_media_5_notas'].mean()
    media_matematica = df_filtrado['nota_mt_matematica'].mean()
    media_redacao = df_filtrado['nota_redacao'].mean()
    total_participantes = len(df_filtrado)

    with col_met1:
        st.metric(label="Total de Participantes (Filtro)", value=f"{total_participantes:,}")
    
    with col_met2:
        st.metric(label="Média Geral (5 Notas)", value=f"{media_geral:.2f} pts")
        
    with col_met3:
        st.metric(label="Média Matemática", value=f"{media_matematica:.2f} pts")
        
    with col_met4:
        st.metric(label="Média Redação", value=f"{media_redacao:.2f} pts")
        
st.markdown("---")

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
