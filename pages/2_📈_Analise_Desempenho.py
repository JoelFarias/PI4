"""
Página 2 - Análise de Desempenho
Análise das notas e distribuições de desempenho no ENEM 2024
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.config import Config
from src.utils.constants import *
from src.database.queries import get_notas_estatisticas, get_participantes_sample
from src.visualization.exploratory import plot_distribution, plot_boxplot


st.set_page_config(
    page_title="Análise de Desempenho - ENEM 2024",
    page_icon="📈",
    layout=Config.APP_LAYOUT
)

st.title("📈 Análise de Desempenho")
st.markdown("Distribuição e estatísticas das notas do ENEM 2024")
st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "📊 Estatísticas Gerais",
    "📉 Distribuições por Prova",
    "🎯 Análise Comparativa"
])

with tab1:
    st.header("📊 Estatísticas Gerais das Notas")
    
    with st.spinner("Carregando estatísticas..."):
        stats = get_notas_estatisticas()
        
        if not stats.empty:
            stats_dict = stats.iloc[0].to_dict()
            
            # Verificar se há valores None e substituir por 0
            for key in stats_dict:
                if stats_dict[key] is None:
                    stats_dict[key] = 0.0
            
            st.subheader("Métricas Principais")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "📊 Média Geral",
                    f"{stats_dict['media_geral']:.1f}",
                    help="Média das 5 notas"
                )
            
            with col2:
                st.metric(
                    "🔬 Ciências da Natureza",
                    f"{stats_dict['cn_media']:.1f}",
                    delta=f"{stats_dict['cn_media'] - stats_dict['media_geral']:.1f}"
                )
            
            with col3:
                st.metric(
                    "📚 Ciências Humanas",
                    f"{stats_dict['ch_media']:.1f}",
                    delta=f"{stats_dict['ch_media'] - stats_dict['media_geral']:.1f}"
                )
            
            with col4:
                st.metric(
                    "📝 Linguagens",
                    f"{stats_dict['lc_media']:.1f}",
                    delta=f"{stats_dict['lc_media'] - stats_dict['media_geral']:.1f}"
                )
            
            with col5:
                st.metric(
                    "➗ Matemática",
                    f"{stats_dict['mt_media']:.1f}",
                    delta=f"{stats_dict['mt_media'] - stats_dict['media_geral']:.1f}"
                )
            
            st.markdown("---")
            
            st.subheader("Tabela de Estatísticas Descritivas")
            
            stats_table = pd.DataFrame({
                'Prova': ['CN', 'CH', 'LC', 'MT', 'Redação', 'Média'],
                'Média': [
                    stats_dict['cn_media'],
                    stats_dict['ch_media'],
                    stats_dict['lc_media'],
                    stats_dict['mt_media'],
                    stats_dict['red_media'],
                    stats_dict['media_geral']
                ],
                'Mediana': [
                    stats_dict['cn_mediana'],
                    stats_dict['ch_mediana'],
                    stats_dict['lc_mediana'],
                    stats_dict['mt_mediana'],
                    stats_dict['red_mediana'],
                    stats_dict['media_mediana']
                ],
                'Desvio Padrão': [
                    stats_dict['cn_desvio'],
                    stats_dict['ch_desvio'],
                    stats_dict['lc_desvio'],
                    stats_dict['mt_desvio'],
                    stats_dict['red_desvio'],
                    stats_dict['media_desvio']
                ],
                'Mínimo': [
                    stats_dict['cn_min'],
                    stats_dict['ch_min'],
                    stats_dict['lc_min'],
                    stats_dict['mt_min'],
                    stats_dict['red_min'],
                    stats_dict['media_min']
                ],
                'Máximo': [
                    stats_dict['cn_max'],
                    stats_dict['ch_max'],
                    stats_dict['lc_max'],
                    stats_dict['mt_max'],
                    stats_dict['red_max'],
                    stats_dict['media_max']
                ]
            })
            
            st.dataframe(
                stats_table.style.format({
                    'Média': '{:.1f}',
                    'Mediana': '{:.1f}',
                    'Desvio Padrão': '{:.1f}',
                    'Mínimo': '{:.1f}',
                    'Máximo': '{:.1f}'
                }).background_gradient(cmap='Blues', subset=['Média', 'Mediana']),
                use_container_width=True
            )
            
            st.markdown("---")
            
            st.subheader("Comparação Visual das Médias")
            
            fig = go.Figure()
            
            areas = ['CN', 'CH', 'LC', 'MT', 'Redação']
            medias = [
                stats_dict['cn_media'],
                stats_dict['ch_media'],
                stats_dict['lc_media'],
                stats_dict['mt_media'],
                stats_dict['red_media']
            ]
            
            fig.add_trace(go.Bar(
                x=areas,
                y=medias,
                text=[f"{m:.1f}" for m in medias],
                textposition='auto',
                marker_color=Config.COLOR_PALETTE[:5],
                name='Média'
            ))
            
            fig.add_hline(
                y=stats_dict['media_geral'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Média Geral: {stats_dict['media_geral']:.1f}"
            )
            
            fig.update_layout(
                title="Média por Área de Conhecimento",
                xaxis_title="Área",
                yaxis_title="Média",
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("📉 Distribuições das Notas por Prova")
    
    prova_selecionada = st.selectbox(
        "Selecione a prova:",
        options=list(NOMES_PROVAS.keys()),
        format_func=lambda x: NOMES_PROVAS[x]
    )
    
    with st.spinner("Carregando dados..."):
        df_notas = get_participantes_sample(
            limit=10000,  # Reduzido para evitar problemas de espaço temporário
            columns=[prova_selecionada],
            where_clause=f"{prova_selecionada} IS NOT NULL"
        )
        
        if not df_notas.empty and prova_selecionada in df_notas.columns:
            # Remover valores nulos da coluna
            df_notas_clean = df_notas[df_notas[prova_selecionada].notna()]
            
            if len(df_notas_clean) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Histograma")
                    fig = px.histogram(
                        df_notas_clean,
                        x=prova_selecionada,
                        nbins=50,
                        marginal="box",
                        color_discrete_sequence=[Config.COLOR_PRIMARY]
                    )
                    fig.update_layout(
                        xaxis_title="Nota",
                        yaxis_title="Frequência",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Box Plot")
                    fig = go.Figure()
                    fig.add_trace(go.Box(
                        y=df_notas_clean[prova_selecionada],
                        name=NOMES_PROVAS[prova_selecionada],
                        marker_color=Config.COLOR_PRIMARY,
                        boxmean='sd'
                    ))
                    fig.update_layout(
                        yaxis_title="Nota",
                        height=500,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Estatísticas Descritivas")
                
                desc_stats = df_notas_clean[prova_selecionada].describe()
                
                # Verificar se há estatísticas válidas
                if len(desc_stats) > 0 and 'mean' in desc_stats.index:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Média", f"{desc_stats.loc['mean']:.1f}")
                        st.metric("Desvio Padrão", f"{desc_stats.loc['std']:.1f}")
                    
                    with col2:
                        st.metric("Mínimo", f"{desc_stats.loc['min']:.1f}")
                        st.metric("Máximo", f"{desc_stats.loc['max']:.1f}")
                    
                    with col3:
                        st.metric("1º Quartil (Q1)", f"{desc_stats.loc['25%']:.1f}")
                        st.metric("Mediana (Q2)", f"{desc_stats.loc['50%']:.1f}")
                    
                    with col4:
                        st.metric("3º Quartil (Q3)", f"{desc_stats.loc['75%']:.1f}")
                        st.metric("Amplitude", f"{desc_stats.loc['max'] - desc_stats.loc['min']:.1f}")
                else:
                    st.warning("⚠️ Não há dados suficientes para calcular estatísticas descritivas.")
            else:
                st.warning("⚠️ Não há dados válidos para esta prova.")
        else:
            st.error("❌ Erro ao carregar dados. Verifique a conexão com o banco.")

with tab3:
    st.header("🎯 Análise Comparativa")
    
    st.subheader("Box Plot Comparativo - Todas as Provas")
    
    with st.spinner("Carregando dados..."):
        df_todas = get_participantes_sample(
            limit=5000,  # Reduzido para evitar problemas de espaço temporário
            columns=COLUNAS_NOTAS[:-1],  # Todas exceto nota_media_5_notas
            where_clause="nota_media_5_notas IS NOT NULL"
        )
        
        if not df_todas.empty:
            # Verificar quais colunas realmente existem
            colunas_disponiveis = [col for col in COLUNAS_NOTAS[:-1] if col in df_todas.columns]
            
            if len(colunas_disponiveis) > 0:
                fig = go.Figure()
                
                for i, col in enumerate(colunas_disponiveis):
                    fig.add_trace(go.Box(
                        y=df_todas[col].dropna(),
                        name=SIGLAS_PROVAS[col],
                        marker_color=Config.COLOR_PALETTE[i % len(Config.COLOR_PALETTE)]
                    ))
                
                fig.update_layout(
                    title="Distribuição de Notas por Prova",
                    yaxis_title="Nota",
                    height=600,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                st.subheader("Violin Plot Comparativo")
                
                fig = go.Figure()
                
                for i, col in enumerate(colunas_disponiveis):
                    fig.add_trace(go.Violin(
                        y=df_todas[col].dropna(),
                        name=SIGLAS_PROVAS[col],
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=Config.COLOR_PALETTE[i % len(Config.COLOR_PALETTE)],
                        opacity=0.6
                    ))
                
                fig.update_layout(
                    title="Distribuição Detalhada - Violin Plot",
                    yaxis_title="Nota",
                    height=600,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                st.subheader("Matriz de Correlação entre Provas")
                
                corr_matrix = df_todas[colunas_disponiveis].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlação"),
                    x=[SIGLAS_PROVAS[col] for col in colunas_disponiveis],
                    y=[SIGLAS_PROVAS[col] for col in colunas_disponiveis],
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1,
                    text_auto='.2f'
                )
                
                fig.update_layout(
                    title="Correlação de Pearson entre as Provas",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 **Interpretação:** Valores próximos de 1 indicam forte correlação positiva, valores próximos de -1 indicam forte correlação negativa, e valores próximos de 0 indicam fraca correlação.")
            else:
                st.warning("⚠️ Nenhuma coluna de notas encontrada nos dados retornados.")
        else:
            st.error("❌ Não foi possível carregar os dados. Verifique a conexão com o banco.")
