"""
Página 3 - Análise de Correlação entre Fatores Familiares e Desempenho

Análise de Correlação Ecológica (Agregada por Município)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from src.database.connection import DatabaseConnection
from src.utils.config import Config

st.set_page_config(
    page_title="Análise de Correlação - ENEM 2024",
    page_icon="🔗",
    layout=Config.APP_LAYOUT
)

st.title("🔗 Análise de Correlação entre Fatores Familiares e Desempenho")

st.info("""
**Metodologia: Correlação Ecológica (Agregada por Município)**

Esta análise examina a relação entre variáveis no nível agregado (municipal), 
não no nível individual. Os dados são agregados por município.

**Limitações:**
- Correlações agregadas não refletem necessariamente correlações individuais
- Perda de variabilidade intra-municipal
- Menor poder estatístico

**Validade:**
- Abordagem aceita em estudos socioeconômicos
- Útil para identificar desigualdades regionais
""")

@st.cache_data(ttl=3600)
def get_dados_agregados_municipio():
    """Agrega dados por município"""
    try:
        query_socio = """
        SELECT 
            co_municipio_prova,
            COUNT(*) as total_alunos,
            ROUND(100.0 * SUM(CASE WHEN q001 IN ('G', 'H') THEN 1 ELSE 0 END) / COUNT(*), 2) as perc_pai_superior,
            ROUND(100.0 * SUM(CASE WHEN q002 IN ('G', 'H') THEN 1 ELSE 0 END) / COUNT(*), 2) as perc_mae_superior,
            ROUND(100.0 * SUM(CASE WHEN q001 IN ('G', 'H') OR q002 IN ('G', 'H') THEN 1 ELSE 0 END) / COUNT(*), 2) as perc_pais_superior,
            ROUND(100.0 * SUM(CASE WHEN q003 IN ('A', 'B') THEN 1 ELSE 0 END) / COUNT(*), 2) as perc_pai_qualificado,
            ROUND(100.0 * SUM(CASE WHEN q004 IN ('A', 'B') THEN 1 ELSE 0 END) / COUNT(*), 2) as perc_mae_qualificado,
            ROUND(100.0 * SUM(CASE WHEN q003 IN ('A', 'B') OR q004 IN ('A', 'B') THEN 1 ELSE 0 END) / COUNT(*), 2) as perc_pais_qualificado,
            ROUND(100.0 * SUM(CASE WHEN q006 IN ('Q', 'P', 'O', 'N', 'M') THEN 1 ELSE 0 END) / COUNT(*), 2) as perc_renda_alta
        FROM ed_enem_2024_participantes
        WHERE co_municipio_prova IS NOT NULL
        GROUP BY co_municipio_prova
        HAVING COUNT(*) >= 30
        """
        
        query_desemp = """
        SELECT 
            co_municipio_prova,
            ROUND(AVG(nota_cn_ciencias_da_natureza)::numeric, 2) as media_cn,
            ROUND(AVG(nota_ch_ciencias_humanas)::numeric, 2) as media_ch,
            ROUND(AVG(nota_lc_linguagens_e_codigos)::numeric, 2) as media_lc,
            ROUND(AVG(nota_mt_matematica)::numeric, 2) as media_mt,
            ROUND(AVG(nota_redacao)::numeric, 2) as media_redacao,
            ROUND(AVG((
                COALESCE(nota_cn_ciencias_da_natureza, 0) +
                COALESCE(nota_ch_ciencias_humanas, 0) +
                COALESCE(nota_lc_linguagens_e_codigos, 0) +
                COALESCE(nota_mt_matematica, 0) +
                COALESCE(nota_redacao, 0)
            ) / 5.0)::numeric, 2) as media_geral
        FROM ed_enem_2024_resultados
        WHERE co_municipio_prova IS NOT NULL
            AND nota_cn_ciencias_da_natureza IS NOT NULL
            AND nota_ch_ciencias_humanas IS NOT NULL
            AND nota_lc_linguagens_e_codigos IS NOT NULL
            AND nota_mt_matematica IS NOT NULL
            AND nota_redacao IS NOT NULL
        GROUP BY co_municipio_prova
        HAVING COUNT(*) >= 30
        """
        
        with DatabaseConnection.get_connection() as conn:
            df_socio = pd.read_sql(query_socio, conn)
            df_desemp = pd.read_sql(query_desemp, conn)
        
        df_completo = pd.merge(df_socio, df_desemp, on='co_municipio_prova', how='inner')
        
        for col in df_completo.columns:
            if df_completo[col].dtype == 'object':
                try:
                    df_completo[col] = pd.to_numeric(df_completo[col], errors='ignore')
                except:
                    pass
        
        return df_completo.astype(float, errors='ignore')
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return pd.DataFrame()

def calcular_correlacao(x, y, nome_x, nome_y):
    """Calcula correlação de Pearson"""
    mask = ~(pd.isna(x) | pd.isna(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 10:
        return None, None, "Dados insuficientes"
    
    corr, p_value = stats.pearsonr(x_clean, y_clean)
    
    if p_value < 0.001:
        sig = "***"
        sig_text = "altamente significativa"
    elif p_value < 0.01:
        sig = "**"
        sig_text = "muito significativa"
    elif p_value < 0.05:
        sig = "*"
        sig_text = "significativa"
    else:
        sig = "ns"
        sig_text = "não significativa"
    
    texto = f"**Correlação:** r = {corr:.3f}{sig} (p = {p_value:.4f}) - {sig_text}"
    return corr, p_value, texto

with st.spinner("Carregando dados..."):
    df = get_dados_agregados_municipio()

if df.empty:
    st.error("Erro ao carregar dados")
    st.stop()

st.success(f"Dados carregados: **{len(df):,}** municípios")

# Debug: Mostrar informações sobre os dados
with st.expander("🔍 Debug - Informações dos Dados"):
    st.write(f"Total de municípios: {len(df)}")
    st.write(f"Colunas: {list(df.columns)}")
    st.write("Primeiras linhas:")
    st.dataframe(df.head())
    st.write("Estatísticas básicas:")
    st.dataframe(df.describe())

tab1, tab2, tab3, tab4 = st.tabs([
    "📚 Escolaridade",
    "💼 Ocupação",
    "💰 Renda",
    "📊 Análise Completa"
])

with tab1:
    st.header("📚 Escolaridade dos Pais vs Desempenho")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pai com Ensino Superior")
        corr_pai, p_pai, texto_pai = calcular_correlacao(
            df['perc_pai_superior'], df['media_geral'], "Pai Superior", "Média"
        )
        st.markdown(texto_pai)
        
        fig_pai = px.scatter(
            df, x='perc_pai_superior', y='media_geral',
            labels={'perc_pai_superior': '% Pais Ens. Superior', 'media_geral': 'Nota Média'},
            title=f"Correlação: {corr_pai:.3f}" if corr_pai else "Correlação: N/A"
        )
        fig_pai.update_traces(marker=dict(size=8, opacity=0.6))
        st.plotly_chart(fig_pai, use_container_width=True)
    
    with col2:
        st.subheader("Mãe com Ensino Superior")
        corr_mae, p_mae, texto_mae = calcular_correlacao(
            df['perc_mae_superior'], df['media_geral'], "Mãe Superior", "Média"
        )
        st.markdown(texto_mae)
        
        fig_mae = px.scatter(
            df, x='perc_mae_superior', y='media_geral',
            labels={'perc_mae_superior': '% Mães Ens. Superior', 'media_geral': 'Nota Média'},
            title=f"Correlação: {corr_mae:.3f}" if corr_mae else "Correlação: N/A"
        )
        fig_mae.update_traces(marker=dict(size=8, opacity=0.6))
        st.plotly_chart(fig_mae, use_container_width=True)

with tab2:
    st.header("💼 Ocupação dos Pais vs Desempenho")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pai Ocupação Qualificada")
        corr_ocup_pai, p_ocup_pai, texto_ocup_pai = calcular_correlacao(
            df['perc_pai_qualificado'], df['media_geral'], "Pai Qualif.", "Média"
        )
        st.markdown(texto_ocup_pai)
        
        fig = px.scatter(
            df, x='perc_pai_qualificado', y='media_geral',
            labels={'perc_pai_qualificado': '% Pais Ocupação Qualif.', 'media_geral': 'Nota Média'},
            title=f"Correlação: {corr_ocup_pai:.3f}" if corr_ocup_pai else "Correlação: N/A"
        )
        fig.update_traces(marker=dict(size=8, opacity=0.6))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Mãe Ocupação Qualificada")
        corr_ocup_mae, p_ocup_mae, texto_ocup_mae = calcular_correlacao(
            df['perc_mae_qualificado'], df['media_geral'], "Mãe Qualif.", "Média"
        )
        st.markdown(texto_ocup_mae)
        
        fig = px.scatter(
            df, x='perc_mae_qualificado', y='media_geral',
            labels={'perc_mae_qualificado': '% Mães Ocupação Qualif.', 'media_geral': 'Nota Média'},
            title=f"Correlação: {corr_ocup_mae:.3f}" if corr_ocup_mae else "Correlação: N/A"
        )
        fig.update_traces(marker=dict(size=8, opacity=0.6))
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("💰 Renda Familiar vs Desempenho")
    
    corr_renda, p_renda, texto_renda = calcular_correlacao(
        df['perc_renda_alta'], df['media_geral'], "Renda Alta", "Média"
    )
    st.markdown(texto_renda)
    
    fig = px.scatter(
        df, x='perc_renda_alta', y='media_geral',
        labels={'perc_renda_alta': '% Famílias Renda Alta', 'media_geral': 'Nota Média'},
        title=f"Correlação: {corr_renda:.3f}" if corr_renda else "Correlação: N/A",
        hover_data=['total_alunos']
    )
    fig.update_traces(marker=dict(size=10, opacity=0.6))
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("📊 Matriz de Correlação")
    
    variaveis_socio = [
        'perc_pai_superior', 'perc_mae_superior',
        'perc_pai_qualificado', 'perc_mae_qualificado', 'perc_renda_alta'
    ]
    
    variaveis_desemp = [
        'media_cn', 'media_ch', 'media_lc', 'media_mt', 'media_redacao', 'media_geral'
    ]
    
    df_corr = df[variaveis_socio + variaveis_desemp].corr()
    matriz = df_corr.loc[variaveis_socio, variaveis_desemp]
    
    rename = {
        'perc_pai_superior': 'Pai Sup.', 'perc_mae_superior': 'Mãe Sup.',
        'perc_pai_qualificado': 'Pai Qualif.', 'perc_mae_qualificado': 'Mãe Qualif.',
        'perc_renda_alta': 'Renda Alta',
        'media_cn': 'CN', 'media_ch': 'CH', 'media_lc': 'LC',
        'media_mt': 'MT', 'media_redacao': 'Red.', 'media_geral': 'Geral'
    }
    
    matriz.rename(index=rename, columns=rename, inplace=True)
    
    fig = px.imshow(
        matriz, text_auto='.3f', aspect='auto',
        color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
        title="Matriz de Correlação: Fatores Socioeconômicos x Desempenho"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Ranking: Correlações com Nota Média Geral")
    
    ranking = []
    for var in variaveis_socio:
        corr, p, _ = calcular_correlacao(df[var], df['media_geral'], var, 'Geral')
        ranking.append({
            'Fator': rename.get(var, var),
            'Correlação': corr,
            'P-value': p,
            'Sig.': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        })
    
    df_ranking = pd.DataFrame(ranking).sort_values('Correlação', ascending=False)
    st.dataframe(
        df_ranking.style.background_gradient(subset=['Correlação'], cmap='RdYlGn'),
        use_container_width=True
    )

st.markdown("---")
st.caption("Nota: Análise de correlação ecológica (nível municipal)")
