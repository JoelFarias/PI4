import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import plotly.express as px
import plotly.graph_objects as go
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
    'H': 'De R$ 5.280,01 a R$ R$ 6.600,00', 'I': 'De R$ 6.600,01 a R$ 7.920,00', 
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
def load_data(sample_size=50000):
    try:
        connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        engine = create_engine(connection_string, pool_pre_ping=True)
        
        # CORREÇÃO: CONSULTA COM INNER JOIN NA TABELA MUNICIPIO
        # Assume-se que 'codigo_municipio_dv' é a chave de ligação na tabela consolidada.
        query = f"""
            SELECT 
                t1.q001 as escolaridade_pai,
                t1.q002 as escolaridade_mae,
                t1.q005 as faixa_renda,
                t1.tp_sexo as sexo,
                t1.tp_cor_raca as cor_raca,
                t1.idade_calculada as idade,
                t1.sg_uf_prova as uf,
                t1.regiao_nome_prova as regiao,
                t2.nome_municipio,
                t1.nota_cn_ciencias_da_natureza,
                t1.nota_ch_ciencias_humanas,
                t1.nota_lc_linguagens_e_codigos,
                t1.nota_mt_matematica,
                t1.nota_redacao,
                t1.nota_media_5_notas
            FROM ed_enem_2024_resultados_amos_per t1
            INNER JOIN municipio t2
                ON t1.codigo_municipio_dv = t2.codigo_municipio_dv
            WHERE t1.q001 IS NOT NULL 
              AND t1.q002 IS NOT NULL 
              AND t1.q005 IS NOT NULL
            ORDER BY RANDOM()
            LIMIT {sample_size};
        """
        df = pd.read_sql(query, engine)
        engine.dispose()
        logging.info(f"Carga com JOIN na tabela municipio: {len(df)} registros carregados.")
        return df
    except SQLAlchemyError as e:
        error_message = f"🚨 Erro SQL ao carregar dados. Verifique o nome das colunas (ex: 'codigo_municipio_dv') ou da tabela 'municipio'. Detalhes: {e}"
        logging.error(error_message)
        st.error(error_message)
        return pd.DataFrame()
    except Exception as e:
        error_message = f"❌ Erro geral ao carregar dados. Verifique a conexão/credenciais. Detalhes: {e}"
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

def create_parent_education_vs_mean_note(df):
    df_agg = df.groupby(['escolaridade_pai', 'escolaridade_mae'])['nota_media_5_notas'].mean().reset_index()
    df_agg.rename(columns={'nota_media_5_notas': 'Nota Média'}, inplace=True)
    
    fig = px.scatter(
        df_agg,
        x='escolaridade_pai',
        y='escolaridade_mae',
        size='Nota Média',
        color='Nota Média',
        title='Média das Notas vs. Escolaridade dos Pais',
        labels={'escolaridade_pai': 'Escolaridade do Pai', 'escolaridade_mae': 'Escolaridade da Mãe'},
        color_continuous_scale=px.colors.sequential.Plasma
    )
    fig.update_xaxes(tickangle=45)
    return fig

def perform_predictive_analysis(df: pd.DataFrame):
    target = 'nota_media_5_notas'
    features = ['faixa_renda', 'escolaridade_pai', 'escolaridade_mae', 'cor_raca', 'sexo']
    
    # 1. Filtragem para garantir dados válidos para o ML
    df_ml = df.dropna(subset=features + [target]).copy()
    df_ml = df_ml[df_ml['nota_media_5_notas'] > 0] # Remove notas zeradas/inválidas

    if len(df_ml) < 100:
        return 0, pd.DataFrame(columns=['Variável', 'Importância', 'Tipo'])

    X = df_ml[features]
    y = df_ml[target]

    # ... Restante da lógica do Random Forest Regressor ...
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features)
        ],
        remainder='passthrough'
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
    model.fit(X_train_processed, y_train)

    y_pred = model.predict(X_test_processed)
    r2 = r2_score(y_test, y_pred)
    
    feature_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(features)
    
    importance_df = pd.DataFrame({
        'Variável': feature_names,
        'Importância': model.feature_importances_
    })
    
    def get_legible_name(var_code):
        parts = var_code.split('_')
        if parts[0] == 'faixa':
            return Q005_MAP.get(parts[-1], f"Renda {parts[-1]}")
        elif parts[0] == 'escolaridade':
            parent_tag = 'Mãe' if 'mae' in var_code else 'Pai'
            return f"{parent_tag}: {escolaridade_map.get(parts[-1], parts[-1])}"
        elif parts[0] == 'cor':
            return f"Cor/Raça: {parts[-1]}"
        elif parts[0] == 'sexo':
            return f"Sexo: {'Masculino' if parts[-1] == 'M' else 'Feminino'}"
        return var_code

    importance_df['Variável'] = importance_df['Variável'].apply(get_legible_name)
    importance_df['Tipo'] = importance_df['Variável'].apply(lambda x: x.split(':')[0].strip())
    
    return r2, importance_df.sort_values(by='Importância', ascending=False).reset_index(drop=True)

st.set_page_config(page_title="ENEM 2024 - Dashboard Socioeconômico", layout="wide")
st.title("📊 ENEM 2024 - Dashboard Socioeconômico")
st.markdown("---")

df_raw = load_data()

if df_raw.empty:
    st.error("⚠️ O DataFrame está vazio. O problema está na origem dos dados. A consulta com JOIN na tabela 'municipio' falhou. Verifique se a tabela 'municipio' existe e se a coluna 'codigo_municipio_dv' é a chave correta para o JOIN.")
    st.stop()

df = decode_enem_categories(df_raw.copy())

# 2. Filtragem de notas nulas e inválidas no Pandas, onde é mais seguro
df_filtrado_notas = df.dropna(subset=['nota_media_5_notas']).copy()
df_filtrado_notas = df_filtrado_notas[df_filtrado_notas['nota_media_5_notas'] > 0]
df_filtrado_notas = df_filtrado_notas.rename(columns={'nota_media_5_notas': 'nota_media_valida'})

st.success(f"✅ Dados carregados e processados: **{len(df):,}** (Brutos) / **{len(df_filtrado_notas):,}** (Com Nota Válida) registros na amostra!")

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

df_filtrado = df_filtrado.dropna(subset=['nota_media_5_notas'])
df_filtrado = df_filtrado[df_filtrado['nota_media_5_notas'] > 0]

st.info(f"Filtros aplicados. Registros para análise (Com Nota Válida): **{len(df_filtrado):,}**")
st.markdown("---")

if len(df_filtrado) == 0:
    st.warning("⚠️ Nenhum registro encontrado com os filtros selecionados ou após a remoção de notas nulas/zero. Tente expandir sua seleção.")
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
    
    fig_parent_notes = create_parent_education_vs_mean_note(df_filtrado)
    st.plotly_chart(fig_parent_notes, use_container_width=True)

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
        O modelo **Random Forest Regressor** foi treinado para prever a **Nota Média (Target)** usando Renda, Escolaridade dos Pais, Cor/Raça e Sexo.
        O relatório abaixo mostra a **Importância das Variáveis**, indicando quais fatores têm maior influência na previsão da nota.
    """)

    r2, importance_df = perform_predictive_analysis(df_filtrado)

    if r2 == 0:
        st.warning("⚠️ Dados insuficientes (menos de 100 registros) para treinar o modelo preditivo. O filtro atual resultou em poucos dados válidos para ML.")
    else:
        col_r2, col_samples = st.columns(2)
        with col_r2:
            st.metric(
                label="Coeficiente de Determinação ($R^2$)", 
                value=f"{r2:.4f}",
                help="Proporção da variância total das notas explicada pelas variáveis socioeconômicas no modelo. Valores mais próximos de 1 são melhores."
            )
        with col_samples:
             st.metric(
                label="Amostra para Predição", 
                value=f"{len(df_filtrado):,}",
                help="O modelo foi treinado com o subset de dados filtrado."
            )

        st.markdown("### Relatório Analítico: Importância das Variáveis")
        st.markdown("""
            **Interpretação:** A **Importância** (soma total = 1.0) mede o quanto uma variável contribuiu para a precisão do modelo Random Forest. Valores mais altos indicam maior influência no desempenho do aluno.
        """)

        fig_importance = px.bar(
            importance_df.head(20),
            x='Importância',
            y='Variável',
            color='Tipo',
            orientation='h',
            title='Top 20 Fatores Socioeconômicos Mais Importantes (Random Forest)',
            height=600
        )
        st.plotly_chart(fig_importance, use_container_width=True)

st.markdown("---")
with st.expander("📄 Ver Dados Brutos Filtrados"):
    st.dataframe(df_filtrado, use_container_width=True)

st.caption("Dashboard ENEM 2024 - Desenvolvido em Python/Streamlit. Dados: PostgreSQL.")
