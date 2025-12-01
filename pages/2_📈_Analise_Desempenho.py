"""
Página 2 - Análise de Desempenho
Análise das médias de desempenho MUNICIPAIS no ENEM 2024

IMPORTANTE: Análise em nível MUNICIPAL (análise ecológica).
Dados agregados por município, não por participante individual.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from src.utils.config import Config
from src.utils.constants import *
from src.utils.theme import apply_minimal_theme, get_plotly_theme
from src.data.loader import load_municipio_data


st.set_page_config(
    page_title="Análise de Desempenho - ENEM 2024",
    page_icon="📈",
    layout=Config.APP_LAYOUT
)

apply_minimal_theme()

st.title("Análise de Desempenho Municipal")
st.markdown("Distribuição e estatísticas das **médias municipais** de notas do ENEM 2024")

# Sidebar com controles
with st.sidebar:
    st.subheader("⚙️ Configurações")
    min_participantes = st.slider(
        "Mínimo de participantes por município",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="Municípios com menos participantes serão excluídos"
    )
    
    if st.button("🔄 Limpar Cache e Recarregar", help="Limpa cache e recarrega dados do banco"):
        st.cache_data.clear()
        st.rerun()

# Aviso sobre análise municipal
st.info("""
**Análise Municipal (Ecológica)**: Esta página analisa as **médias de desempenho por município**, 
não notas individuais. Cada ponto representa um município (~5.570 municípios no Brasil).
""")

st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "📊 Estatísticas Municipais",
    "📉 Distribuições por Prova",
    "🎯 Comparação entre Provas"
])

with tab1:
    st.header("📊 Estatísticas das Médias Municipais")
    st.markdown("Estatísticas calculadas sobre as **médias municipais** de cada prova")
    
    with st.spinner("Carregando dados municipais..."):
        df_municipios = load_municipio_data(min_participantes=min_participantes)
        
        if not df_municipios.empty:
            # Calcular estatísticas das médias municipais
            stats_municipais = {
                'total_municipios': len(df_municipios),
                'cn_media': df_municipios['media_cn'].mean(),
                'cn_std': df_municipios['media_cn'].std(),
                'ch_media': df_municipios['media_ch'].mean(),
                'ch_std': df_municipios['media_ch'].std(),
                'lc_media': df_municipios['media_lc'].mean(),
                'lc_std': df_municipios['media_lc'].std(),
                'mt_media': df_municipios['media_mt'].mean(),
                'mt_std': df_municipios['media_mt'].std(),
                'red_media': df_municipios['media_redacao'].mean(),
                'red_std': df_municipios['media_redacao'].std(),
                'geral_media': df_municipios['media_geral'].mean(),
                'geral_std': df_municipios['media_geral'].std()
            }
            
            st.subheader("Métricas Principais (Médias entre Municípios)")
            st.caption(f"Baseado em {stats_municipais['total_municipios']:,.0f} municípios")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "📊 Média Geral",
                    f"{stats_municipais['geral_media']:.1f}",
                    help=f"Média entre todos os municípios. σ = {stats_municipais['geral_std']:.1f}"
                )
            
            with col2:
                st.metric(
                    "🔬 Ciências da Natureza",
                    f"{stats_municipais['cn_media']:.1f}",
                    delta=f"{stats_municipais['cn_media'] - stats_municipais['geral_media']:.1f}"
                )
            
            with col3:
                st.metric(
                    "📚 Ciências Humanas",
                    f"{stats_municipais['ch_media']:.1f}",
                    delta=f"{stats_municipais['ch_media'] - stats_municipais['geral_media']:.1f}"
                )
            
            with col4:
                st.metric(
                    "📝 Linguagens",
                    f"{stats_municipais['lc_media']:.1f}",
                    delta=f"{stats_municipais['lc_media'] - stats_municipais['geral_media']:.1f}"
                )
            
            with col5:
                st.metric(
                    "➗ Matemática",
                    f"{stats_municipais['mt_media']:.1f}",
                    delta=f"{stats_municipais['mt_media'] - stats_municipais['geral_media']:.1f}"
                )
            
            st.markdown("---")
            
            st.subheader("Tabela de Estatísticas Descritivas (Municípios)")
            
            stats_table = pd.DataFrame({
                'Prova': ['CN', 'CH', 'LC', 'MT', 'Redação', 'Média Geral'],
                'Média': [
                    stats_municipais['cn_media'],
                    stats_municipais['ch_media'],
                    stats_municipais['lc_media'],
                    stats_municipais['mt_media'],
                    stats_municipais['red_media'],
                    stats_municipais['geral_media']
                ],
                'Desvio Padrão': [
                    stats_municipais['cn_std'],
                    stats_municipais['ch_std'],
                    stats_municipais['lc_std'],
                    stats_municipais['mt_std'],
                    stats_municipais['red_std'],
                    stats_municipais['geral_std']
                ],
                'Mínimo': [
                    df_municipios['media_cn'].min(),
                    df_municipios['media_ch'].min(),
                    df_municipios['media_lc'].min(),
                    df_municipios['media_mt'].min(),
                    df_municipios['media_redacao'].min(),
                    df_municipios['media_geral'].min()
                ],
                'Máximo': [
                    df_municipios['media_cn'].max(),
                    df_municipios['media_ch'].max(),
                    df_municipios['media_lc'].max(),
                    df_municipios['media_mt'].max(),
                    df_municipios['media_redacao'].max(),
                    df_municipios['media_geral'].max()
                ]
            })
            
            st.dataframe(
                stats_table.style.format({
                    'Média': '{:.1f}',
                    'Desvio Padrão': '{:.1f}',
                    'Mínimo': '{:.1f}',
                    'Máximo': '{:.1f}'
                }).background_gradient(cmap='Blues', subset=['Média']),
                use_container_width=True
            )
            
            st.markdown("---")
            
            st.subheader("Comparação Visual das Médias")
            
            fig = go.Figure()
            
            areas = ['CN', 'CH', 'LC', 'MT', 'Redação']
            medias = [
                float(stats_municipais['cn_media']),
                float(stats_municipais['ch_media']),
                float(stats_municipais['lc_media']),
                float(stats_municipais['mt_media']),
                float(stats_municipais['red_media'])
            ]
            
            fig.add_trace(go.Bar(
                x=areas,
                y=medias,
                text=[f"{m:.1f}" for m in medias],
                textposition='auto',
                marker_color=Config.COLOR_PALETTE[:5],
                hovertemplate='<b>%{x}</b><br>Média: %{y:.1f}<extra></extra>'
            ))
            
            fig.add_hline(
                y=stats_municipais['geral_media'],
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"Média Geral: {stats_municipais['geral_media']:.1f}"
            )
            
            theme = get_plotly_theme()
            fig.update_layout(
                **{k: v for k, v in theme.items() if k not in ['xaxis', 'yaxis', 'margin']},
                xaxis=dict(title="Área de Conhecimento", tickangle=0),
                yaxis=dict(title="Média Municipal", tickformat=".1f"),
                height=450,
                showlegend=False,
                margin=dict(t=40, b=60, l=60, r=30)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Não foi possível carregar os dados municipais.")

with tab2:
    st.header("📉 Distribuições das Médias Municipais por Prova")
    
    # Mapear colunas do DataFrame para nomes das provas
    mapa_colunas = {
        'nu_nota_cn': 'media_cn',
        'nu_nota_ch': 'media_ch',
        'nu_nota_lc': 'media_lc',
        'nu_nota_mt': 'media_mt',
        'nu_nota_redacao': 'media_redacao'
    }
    
    prova_selecionada = st.selectbox(
        "Selecione a prova:",
        options=list(NOMES_PROVAS.keys()),
        format_func=lambda x: NOMES_PROVAS[x],
        key="selectbox_tab2_prova"
    )
    
    # Obter o nome da coluna municipal correspondente
    coluna_municipal = mapa_colunas.get(prova_selecionada)
    
    if coluna_municipal:
        with st.spinner("Carregando dados municipais..."):
            df_municipios = load_municipio_data(min_participantes=min_participantes)
            
            if not df_municipios.empty and coluna_municipal in df_municipios.columns:
                df_prova = df_municipios[coluna_municipal].dropna()
                
                if len(df_prova) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Histograma das Médias Municipais")
                        fig = px.histogram(
                            df_prova,
                            nbins=50,
                            color_discrete_sequence=[Config.COLOR_PRIMARY]
                        )
                        theme = get_plotly_theme()
                        fig.update_layout(
                            **{k: v for k, v in theme.items() if k not in ['xaxis', 'yaxis', 'margin']},
                            xaxis=dict(title="Média Municipal", tickformat=".0f"),
                            yaxis=dict(title="Número de Municípios"),
                            height=400,
                            showlegend=False,
                            margin=dict(t=30, b=60, l=60, r=30)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("Box Plot")
                        fig = go.Figure()
                        fig.add_trace(go.Box(
                            y=df_prova,
                            name=NOMES_PROVAS[prova_selecionada],
                            marker_color=Config.COLOR_PRIMARY,
                            boxmean='sd'
                        ))
                        theme = get_plotly_theme()
                        fig.update_layout(
                            **{k: v for k, v in theme.items() if k not in ['yaxis', 'margin']},
                            yaxis=dict(title="Média Municipal", tickformat=".0f"),
                            height=400,
                            showlegend=False,
                            margin=dict(t=30, b=60, l=60, r=30)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Estatísticas Descritivas")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Média", f"{df_prova.mean():.1f}")
                        st.metric("Desvio Padrão", f"{df_prova.std():.1f}")
                    
                    with col2:
                        st.metric("Mínimo", f"{df_prova.min():.1f}")
                        st.metric("Máximo", f"{df_prova.max():.1f}")
                    
                    with col3:
                        st.metric("1º Quartil (Q1)", f"{df_prova.quantile(0.25):.1f}")
                        st.metric("Mediana (Q2)", f"{df_prova.median():.1f}")
                    
                    with col4:
                        st.metric("3º Quartil (Q3)", f"{df_prova.quantile(0.75):.1f}")
                        amplitude = df_prova.max() - df_prova.min()
                        st.metric("Amplitude", f"{amplitude:.1f}")
                else:
                    st.warning("⚠️ Não há dados válidos para esta prova.")
            else:
                st.error("❌ Coluna não encontrada nos dados municipais.")
    else:
        st.error("❌ Mapeamento de coluna não encontrado.")

with tab3:
    st.header("🎯 Comparação entre Provas")
    
    st.subheader("Box Plot Comparativo - Todas as Provas")
    
    with st.spinner("Carregando dados municipais..."):
        df_municipios = load_municipio_data(min_participantes=min_participantes)
        
        if not df_municipios.empty:
            colunas_notas = ['media_cn', 'media_ch', 'media_lc', 'media_mt', 'media_redacao']
            nomes_provas = ['CN', 'CH', 'LC', 'MT', 'Redação']
            
            # Verificar colunas disponíveis
            colunas_disponiveis = [col for col in colunas_notas if col in df_municipios.columns]
            
            if colunas_disponiveis:
                fig = go.Figure()
                
                for i, col in enumerate(colunas_disponiveis):
                    fig.add_trace(go.Box(
                        y=df_municipios[col].dropna(),
                        name=nomes_provas[i],
                        marker_color=Config.COLOR_PALETTE[i % len(Config.COLOR_PALETTE)],
                        hovertemplate='<b>%{fullData.name}</b><br>Média: %{y:.1f}<extra></extra>'
                    ))
                
                theme = get_plotly_theme()
                fig.update_layout(
                    **{k: v for k, v in theme.items() if k not in ['yaxis', 'margin']},
                    yaxis=dict(title="Média Municipal", tickformat=".0f"),
                    height=500,
                    showlegend=True,
                    margin=dict(t=40, b=60, l=60, r=30)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                st.subheader("Violin Plot Comparativo")
                
                fig = go.Figure()
                
                for i, col in enumerate(colunas_disponiveis):
                    fig.add_trace(go.Violin(
                        y=df_municipios[col].dropna(),
                        name=nomes_provas[i],
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=Config.COLOR_PALETTE[i % len(Config.COLOR_PALETTE)],
                        opacity=0.6,
                        hovertemplate='<b>%{fullData.name}</b><br>Média: %{y:.1f}<extra></extra>'
                    ))
                
                theme = get_plotly_theme()
                fig.update_layout(
                    **{k: v for k, v in theme.items() if k not in ['yaxis', 'margin']},
                    yaxis=dict(title="Média Municipal", tickformat=".0f"),
                    height=500,
                    showlegend=True,
                    margin=dict(t=40, b=60, l=60, r=30)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                st.subheader("Matriz de Correlação entre Provas")
                
                corr_matrix = df_municipios[colunas_disponiveis].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlação"),
                    x=nomes_provas[:len(colunas_disponiveis)],
                    y=nomes_provas[:len(colunas_disponiveis)],
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1,
                    text_auto='.2f'
                )
                
                theme = get_plotly_theme()
                fig.update_layout(
                    **{k: v for k, v in theme.items() if k not in ['margin']},
                    height=500,
                    margin=dict(t=40, b=60, l=60, r=60)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 **Interpretação:** Valores próximos de 1 indicam forte correlação positiva entre as médias municipais das provas. Municípios com boas médias em uma prova tendem a ter boas médias em outra.")
            else:
                st.warning("⚠️ Nenhuma coluna de notas encontrada.")
        else:
            st.error("❌ Não foi possível carregar os dados municipais.")
