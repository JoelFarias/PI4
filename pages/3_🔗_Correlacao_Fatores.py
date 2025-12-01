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
from src.data.loader import load_municipio_data
from src.utils.config import Config
from src.utils.theme import apply_minimal_theme, get_plotly_theme

st.set_page_config(
    page_title="Análise de Correlação - ENEM 2024",
    page_icon="🔗",
    layout=Config.APP_LAYOUT
)

apply_minimal_theme()

st.title("Correlação entre Fatores Familiares e Desempenho")

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
    """Carrega dados agregados por município"""
    try:
        # Usar a função centralizada que já foi corrigida
        df_completo = load_municipio_data(min_participantes=30)
        
        if df_completo.empty:
            return pd.DataFrame()
        
        # Renomear colunas para manter compatibilidade com código existente
        rename_map = {
            'total_participantes': 'total_alunos',
            'perc_pais_ensino_superior': 'perc_pais_superior',
            'perc_pai_ensino_superior': 'perc_pai_superior',
            'perc_mae_ensino_superior': 'perc_mae_superior',
            'perc_pai_ocupacao_qualificada': 'perc_pai_qualificado',
            'perc_mae_ocupacao_qualificada': 'perc_mae_qualificado'
        }
        
        df_completo = df_completo.rename(columns=rename_map)
        
        # Adicionar coluna de pais qualificados combinada se não existir
        if 'perc_pais_qualificado' not in df_completo.columns:
            df_completo['perc_pais_qualificado'] = (
                df_completo['perc_pai_qualificado'] + df_completo['perc_mae_qualificado']
            ) / 2
        
        # Garantir que todas as colunas numéricas sejam do tipo float
        numeric_cols = [
            'total_alunos', 
            'perc_pai_superior', 'perc_mae_superior', 'perc_pais_superior',
            'perc_pai_qualificado', 'perc_mae_qualificado', 'perc_pais_qualificado',
            'perc_renda_alta', 'perc_renda_baixa',
            'media_pessoas_residencia',
            'perc_escola_privada', 'perc_escola_publica',
            'perc_feminino', 'idade_media',
            'media_cn', 'media_ch', 'media_lc', 'media_mt', 'media_redacao', 'media_geral'
        ]
        
        for col in numeric_cols:
            if col in df_completo.columns:
                df_completo[col] = pd.to_numeric(df_completo[col], errors='coerce')
        
        # Remover linhas onde as colunas essenciais são NaN
        essential_cols = ['perc_pai_superior', 'perc_mae_superior', 'media_geral']
        df_completo = df_completo.dropna(subset=[col for col in essential_cols if col in df_completo.columns])
        
        return df_completo
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return pd.DataFrame()

def calcular_correlacao(x, y, nome_x, nome_y):
    """Calcula correlação de Pearson com validação robusta"""
    # Garantir que são Series do pandas
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    # Converter para numérico
    x = pd.to_numeric(x, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    
    # Filtrar NaN e infinitos
    mask = ~(pd.isna(x) | pd.isna(y) | np.isinf(x) | np.isinf(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    # Verificar variância
    if len(x_clean) < 10:
        return None, None, "❌ **Dados insuficientes** (menos de 10 observações válidas)"
    
    if x_clean.std() == 0:
        return None, None, f"❌ **Sem variação em {nome_x}** (todos os valores são iguais)"
    
    if y_clean.std() == 0:
        return None, None, f"❌ **Sem variação em {nome_y}** (todos os valores são iguais)"
    
    try:
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
        texto += f"\n\n📊 **N válido:** {len(x_clean):,} municípios"
        return corr, p_value, texto
        
    except Exception as e:
        return None, None, f"❌ **Erro ao calcular:** {str(e)}"

with st.spinner("⏳ Carregando dados municipais... Isso pode levar 30-60 segundos na primeira vez."):
    df = get_dados_agregados_municipio()

if df.empty:
    st.error("❌ Erro ao carregar dados. Verifique a conexão com o banco de dados.")
    st.stop()

st.success(f"✅ Dados carregados: **{len(df):,}** municípios")

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
        
        if corr_pai is not None:
            fig_pai = px.scatter(
                df, x='perc_pai_superior', y='media_geral',
                labels={'perc_pai_superior': '% Pais Ens. Superior', 'media_geral': 'Nota Média'},
                title=f"Correlação: r = {corr_pai:.3f}",
                trendline="ols",
                hover_data=['municipio', 'uf'] if 'municipio' in df.columns else None
            )
            fig_pai.update_traces(marker=dict(size=8, opacity=0.6, color='steelblue'))
            fig_pai.update_layout(height=400)
            st.plotly_chart(fig_pai, use_container_width=True)
        else:
            st.warning("Não foi possível gerar o gráfico")
    
    with col2:
        st.subheader("Mãe com Ensino Superior")
        corr_mae, p_mae, texto_mae = calcular_correlacao(
            df['perc_mae_superior'], df['media_geral'], "Mãe Superior", "Média"
        )
        st.markdown(texto_mae)
        
        if corr_mae is not None:
            fig_mae = px.scatter(
                df, x='perc_mae_superior', y='media_geral',
                labels={'perc_mae_superior': '% Mães Ens. Superior', 'media_geral': 'Nota Média'},
                title=f"Correlação: r = {corr_mae:.3f}",
                trendline="ols",
                hover_data=['municipio', 'uf'] if 'municipio' in df.columns else None
            )
            fig_mae.update_traces(marker=dict(size=8, opacity=0.6, color='coral'))
            fig_mae.update_layout(height=400)
            st.plotly_chart(fig_mae, use_container_width=True)
        else:
            st.warning("Não foi possível gerar o gráfico")

with tab2:
    st.header("💼 Ocupação dos Pais vs Desempenho")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pai Ocupação Qualificada")
        corr_ocup_pai, p_ocup_pai, texto_ocup_pai = calcular_correlacao(
            df['perc_pai_qualificado'], df['media_geral'], "Pai Qualif.", "Média"
        )
        st.markdown(texto_ocup_pai)
        
        if corr_ocup_pai is not None:
            fig = px.scatter(
                df, x='perc_pai_qualificado', y='media_geral',
                labels={'perc_pai_qualificado': '% Pais Ocupação Qualif.', 'media_geral': 'Nota Média'},
                title=f"Correlação: r = {corr_ocup_pai:.3f}",
                trendline="ols",
                hover_data=['municipio', 'uf'] if 'municipio' in df.columns else None
            )
            fig.update_traces(marker=dict(size=8, opacity=0.6, color='green'))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Não foi possível gerar o gráfico")
    
    with col2:
        st.subheader("Mãe Ocupação Qualificada")
        corr_ocup_mae, p_ocup_mae, texto_ocup_mae = calcular_correlacao(
            df['perc_mae_qualificado'], df['media_geral'], "Mãe Qualif.", "Média"
        )
        st.markdown(texto_ocup_mae)
        
        if corr_ocup_mae is not None:
            fig = px.scatter(
                df, x='perc_mae_qualificado', y='media_geral',
                labels={'perc_mae_qualificado': '% Mães Ocupação Qualif.', 'media_geral': 'Nota Média'},
                title=f"Correlação: r = {corr_ocup_mae:.3f}",
                trendline="ols",
                hover_data=['municipio', 'uf'] if 'municipio' in df.columns else None
            )
            fig.update_traces(marker=dict(size=8, opacity=0.6, color='purple'))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Não foi possível gerar o gráfico")

with tab3:
    st.header("💰 Renda Familiar vs Desempenho")
    
    corr_renda, p_renda, texto_renda = calcular_correlacao(
        df['perc_renda_alta'], df['media_geral'], "Renda Alta", "Média"
    )
    st.markdown(texto_renda)
    
    if corr_renda is not None:
        fig = px.scatter(
            df, x='perc_renda_alta', y='media_geral',
            labels={'perc_renda_alta': '% Famílias Renda Alta (>R$ 16.944)', 'media_geral': 'Nota Média ENEM'},
            title=f"Correlação: r = {corr_renda:.3f}",
            trendline="ols",
            hover_data=['municipio', 'uf', 'total_alunos'] if 'municipio' in df.columns else ['total_alunos']
        )
        fig.update_traces(marker=dict(size=10, opacity=0.6, color='gold'))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Não foi possível gerar o gráfico")

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
            'Correlação': corr if corr is not None else 0.0,
            'P-value': p if p is not None else 1.0,
            'Sig.': '***' if p is not None and p < 0.001 else '**' if p is not None and p < 0.01 else '*' if p is not None and p < 0.05 else 'ns'
        })
    
    df_ranking = pd.DataFrame(ranking).sort_values('Correlação', ascending=False)
    st.dataframe(
        df_ranking.style.background_gradient(subset=['Correlação'], cmap='RdYlGn'),
        use_container_width=True
    )

st.markdown("---")
st.caption("Nota: Análise de correlação ecológica (nível municipal)")
