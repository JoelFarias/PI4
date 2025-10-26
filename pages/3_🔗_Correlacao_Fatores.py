"""
Página 3 - Análise de Correlação entre Fatores Familiares e Desempenho
Foco na questão de pesquisa: correlação entre escolaridade/ocupação dos pais e notas
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, f_oneway

from src.utils.config import Config
from src.utils.constants import *
from src.database.queries import (
    get_participantes_sample,
    get_media_por_escolaridade_pai,
    get_media_por_escolaridade_mae,
    get_media_por_ocupacao_pai,
    get_media_por_ocupacao_mae,
    get_media_por_renda
)
from src.visualization.correlation import (
    plot_correlation_heatmap,
    plot_grouped_boxplot,
    plot_grouped_bar_with_error,
    plot_violin_comparison,
    create_correlation_table,
    plot_correlation_strength
)


st.set_page_config(
    page_title="Análise de Correlação - ENEM 2024",
    page_icon="🔗",
    layout=Config.APP_LAYOUT
)

st.title("🔗 Análise de Correlação")
st.markdown("""
**Questão de Pesquisa:** Qual é a correlação entre o desempenho dos alunos no ENEM 2024 
e o nível de escolaridade e ocupação dos seus pais?
""")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📚 Escolaridade dos Pais",
    "💼 Ocupação dos Pais",
    "💰 Renda Familiar",
    "📊 Matriz de Correlação"
])

with tab1:
    st.header("📚 Escolaridade dos Pais vs Desempenho")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Escolaridade do Pai")
        
        with st.spinner("Carregando dados..."):
            df_pai = get_media_por_escolaridade_pai()
            
            if not df_pai.empty:
                df_pai['escolaridade_pai_label'] = df_pai['escolaridade_pai'].map(ESCOLARIDADE_PAI)
                
                # Filtrar valores nulos e ordenar
                df_pai = df_pai.dropna(subset=['media_geral'])
                df_pai = df_pai.sort_values('media_geral', ascending=False)
                
                st.markdown("##### Média Geral por Nível de Escolaridade")
                
                fig, stats = plot_grouped_bar_with_error(
                    df_pai,
                    'escolaridade_pai_label',
                    'media_geral',
                    title="Média Geral por Escolaridade do Pai"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("##### Estatísticas Detalhadas")
                st.dataframe(
                    stats.rename(columns={
                        'escolaridade_pai_label': 'Escolaridade',
                        'mean': 'Média',
                        'std': 'Desvio Padrão',
                        'count': 'N'
                    }).style.format({
                        'Média': '{:.1f}',
                        'Desvio Padrão': '{:.1f}',
                        'N': '{:.0f}'
                    }).background_gradient(cmap='Blues', subset=['Média']),
                    use_container_width=True
                )
                
                maior_esc = df_pai.iloc[0]['escolaridade_pai_label']
                menor_esc = df_pai.iloc[-1]['escolaridade_pai_label']
                diferenca = df_pai.iloc[0]['media_geral'] - df_pai.iloc[-1]['media_geral']
                
                st.info(f"""
                **💡 Insight:** Estudantes cujos pais têm **{maior_esc}** têm média **{diferenca:.1f} pontos** 
                superior aos cujos pais têm **{menor_esc}**.
                """)
    
    with col2:
        st.subheader("Escolaridade da Mãe")
        
        with st.spinner("Carregando dados..."):
            df_mae = get_media_por_escolaridade_mae()
            
            if not df_mae.empty:
                df_mae['escolaridade_mae_label'] = df_mae['escolaridade_mae'].map(ESCOLARIDADE_MAE)
                
                # Filtrar valores nulos e ordenar
                df_mae = df_mae.dropna(subset=['media_geral'])
                df_mae = df_mae.sort_values('media_geral', ascending=False)
                
                st.markdown("##### Média Geral por Nível de Escolaridade")
                
                fig, stats = plot_grouped_bar_with_error(
                    df_mae,
                    'escolaridade_mae_label',
                    'media_geral',
                    title="Média Geral por Escolaridade da Mãe"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("##### Estatísticas Detalhadas")
                st.dataframe(
                    stats.rename(columns={
                        'escolaridade_mae_label': 'Escolaridade',
                        'mean': 'Média',
                        'std': 'Desvio Padrão',
                        'count': 'N'
                    }).style.format({
                        'Média': '{:.1f}',
                        'Desvio Padrão': '{:.1f}',
                        'N': '{:.0f}'
                    }).background_gradient(cmap='Greens', subset=['Média']),
                    use_container_width=True
                )
                
                maior_esc = df_mae.iloc[0]['escolaridade_mae_label']
                menor_esc = df_mae.iloc[-1]['escolaridade_mae_label']
                diferenca = df_mae.iloc[0]['media_geral'] - df_mae.iloc[-1]['media_geral']
                
                st.info(f"""
                **💡 Insight:** Estudantes cujas mães têm **{maior_esc}** têm média **{diferenca:.1f} pontos** 
                superior aos cujas mães têm **{menor_esc}**.
                """)
    
    st.markdown("---")
    
    st.subheader("📊 Comparação por Área de Conhecimento")
    
    area_selecionada = st.selectbox(
        "Selecione a área:",
        options=[
            ('cn_media', 'Ciências da Natureza'),
            ('ch_media', 'Ciências Humanas'),
            ('lc_media', 'Linguagens e Códigos'),
            ('mt_media', 'Matemática'),
            ('red_media', 'Redação')
        ],
        format_func=lambda x: x[1]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not df_pai.empty:
            fig, _ = plot_grouped_bar_with_error(
                df_pai,
                'escolaridade_pai_label',
                area_selecionada[0],
                value_label=area_selecionada[1],
                title=f"{area_selecionada[1]} - Escolaridade do Pai"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if not df_mae.empty:
            fig, _ = plot_grouped_bar_with_error(
                df_mae,
                'escolaridade_mae_label',
                area_selecionada[0],
                value_label=area_selecionada[1],
                title=f"{area_selecionada[1]} - Escolaridade da Mãe"
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("💼 Ocupação dos Pais vs Desempenho")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ocupação do Pai")
        
        with st.spinner("Carregando dados..."):
            df_ocup_pai = get_media_por_ocupacao_pai()
            
            if not df_ocup_pai.empty:
                df_ocup_pai['ocupacao_pai_label'] = df_ocup_pai['ocupacao_pai'].map(OCUPACAO_PAI)
                df_ocup_pai = df_ocup_pai.sort_values('media_geral', ascending=False)
                
                fig, stats = plot_grouped_bar_with_error(
                    df_ocup_pai,
                    'ocupacao_pai_label',
                    'media_geral',
                    title="Média Geral por Ocupação do Pai"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(
                    stats.rename(columns={
                        'ocupacao_pai_label': 'Ocupação',
                        'mean': 'Média',
                        'std': 'Desvio Padrão',
                        'count': 'N'
                    }).style.format({
                        'Média': '{:.1f}',
                        'Desvio Padrão': '{:.1f}',
                        'N': '{:.0f}'
                    }).background_gradient(cmap='Blues', subset=['Média']),
                    use_container_width=True
                )
    
    with col2:
        st.subheader("Ocupação da Mãe")
        
        with st.spinner("Carregando dados..."):
            df_ocup_mae = get_media_por_ocupacao_mae()
            
            if not df_ocup_mae.empty:
                df_ocup_mae['ocupacao_mae_label'] = df_ocup_mae['ocupacao_mae'].map(OCUPACAO_MAE)
                df_ocup_mae = df_ocup_mae.sort_values('media_geral', ascending=False)
                
                fig, stats = plot_grouped_bar_with_error(
                    df_ocup_mae,
                    'ocupacao_mae_label',
                    'media_geral',
                    title="Média Geral por Ocupação da Mãe"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(
                    stats.rename(columns={
                        'ocupacao_mae_label': 'Ocupação',
                        'mean': 'Média',
                        'std': 'Desvio Padrão',
                        'count': 'N'
                    }).style.format({
                        'Média': '{:.1f}',
                        'Desvio Padrão': '{:.1f}',
                        'N': '{:.0f}'
                    }).background_gradient(cmap='Greens', subset=['Média']),
                    use_container_width=True
                )
    
    st.markdown("---")
    
    st.subheader("📊 Violin Plot - Distribuição Completa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not df_ocup_pai.empty:
            with st.spinner("Carregando dados..."):
                df_sample = get_participantes_sample(
                    limit=5000,  # Reduzido para evitar problemas de espaço
                    columns=['ocupacao_pai', 'nota_media_5_notas'],
                    where_clause="ocupacao_pai IS NOT NULL AND nota_media_5_notas IS NOT NULL"
                )
                
                if not df_sample.empty:
                    fig = plot_violin_comparison(
                        df_sample,
                        'ocupacao_pai',
                        'nota_media_5_notas',
                        category_labels=OCUPACAO_PAI,
                        value_label="Média Geral",
                        title="Distribuição de Notas - Ocupação do Pai"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if not df_ocup_mae.empty:
            with st.spinner("Carregando dados..."):
                df_sample = get_participantes_sample(
                    limit=5000,  # Reduzido para evitar problemas de espaço
                    columns=['ocupacao_mae', 'nota_media_5_notas'],
                    where_clause="ocupacao_mae IS NOT NULL AND nota_media_5_notas IS NOT NULL"
                )
                
                if not df_sample.empty:
                    fig = plot_violin_comparison(
                        df_sample,
                        'ocupacao_mae',
                        'nota_media_5_notas',
                        category_labels=OCUPACAO_MAE,
                        value_label="Média Geral",
                        title="Distribuição de Notas - Ocupação da Mãe"
                    )
                    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("💰 Renda Familiar vs Desempenho")
    
    with st.spinner("Carregando dados..."):
        df_renda = get_media_por_renda()
        
        if not df_renda.empty:
            df_renda['faixa_renda_label'] = df_renda['faixa_renda'].map(FAIXA_RENDA)
            
            faixa_order = list(FAIXA_RENDA.values())
            df_renda['faixa_renda_label'] = pd.Categorical(
                df_renda['faixa_renda_label'],
                categories=faixa_order,
                ordered=True
            )
            df_renda = df_renda.sort_values('faixa_renda_label')
            
            st.subheader("Média Geral por Faixa de Renda")
            
            fig, stats = plot_grouped_bar_with_error(
                df_renda,
                'faixa_renda_label',
                'media_geral',
                title="Desempenho por Faixa de Renda Familiar"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Tabela Detalhada")
            
            st.dataframe(
                stats.rename(columns={
                    'faixa_renda_label': 'Faixa de Renda',
                    'mean': 'Média',
                    'std': 'Desvio Padrão',
                    'count': 'N'
                }).style.format({
                    'Média': '{:.1f}',
                    'Desvio Padrão': '{:.1f}',
                    'N': '{:.0f}'
                }).background_gradient(cmap='YlGn', subset=['Média']),
                use_container_width=True
            )
            
            maior_renda_idx = stats['mean'].idxmax()
            menor_renda_idx = stats['mean'].idxmin()
            diferenca = stats.loc[maior_renda_idx, 'mean'] - stats.loc[menor_renda_idx, 'mean']
            
            st.success(f"""
            **💡 Insight Principal:** A diferença entre a maior e menor faixa de renda é de 
            **{diferenca:.1f} pontos** na média geral.
            """)
            
            st.markdown("---")
            
            st.subheader("📊 Comparação por Área de Conhecimento")
            
            areas = [
                ('cn_media', 'CN'),
                ('ch_media', 'CH'),
                ('lc_media', 'LC'),
                ('mt_media', 'MT'),
                ('red_media', 'Redação')
            ]
            
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=1, cols=5,
                subplot_titles=[nome for _, nome in areas],
                shared_yaxes=True
            )
            
            for idx, (col, nome) in enumerate(areas, 1):
                fig.add_trace(
                    go.Bar(
                        x=df_renda['faixa_renda_label'],
                        y=df_renda[col],
                        name=nome,
                        marker_color=Config.COLOR_PALETTE[idx-1]
                    ),
                    row=1, col=idx
                )
            
            fig.update_layout(
                height=400,
                showlegend=False,
                title_text="Desempenho por Área e Faixa de Renda"
            )
            
            fig.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("📊 Matriz de Correlação Completa")
    
    st.markdown("""
    Análise de correlação entre todas as variáveis socioeconômicas e o desempenho.
    """)
    
    with st.spinner("Carregando amostra de dados..."):
        df_completo = get_participantes_sample(
            limit=10000,  # Reduzido para evitar problemas de espaço
            columns=[
                'escolaridade_pai', 'escolaridade_mae',
                'ocupacao_pai', 'ocupacao_mae',
                'faixa_renda', 'pessoas_residencia',
                'nota_cn_ciencias_da_natureza',
                'nota_ch_ciencias_humanas',
                'nota_lc_linguagens_e_codigos',
                'nota_mt_matematica',
                'nota_redacao',
                'nota_media_5_notas'
            ],
            where_clause="nota_media_5_notas IS NOT NULL"
        )
        
        if not df_completo.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Heatmap de Correlação")
                
                colunas_numericas = [
                    'escolaridade_pai', 'escolaridade_mae',
                    'ocupacao_pai', 'ocupacao_mae',
                    'faixa_renda', 'pessoas_residencia',
                    'nota_cn_ciencias_da_natureza',
                    'nota_ch_ciencias_humanas',
                    'nota_lc_linguagens_e_codigos',
                    'nota_mt_matematica',
                    'nota_redacao',
                    'nota_media_5_notas'
                ]
                
                fig = plot_correlation_heatmap(
                    df_completo,
                    columns=colunas_numericas,
                    method='pearson',
                    title="Correlação de Pearson - Variáveis Socioeconômicas e Notas"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Interpretação")
                
                st.markdown("""
                **Escala de Correlação:**
                - 🟢 **0.7 a 1.0**: Forte positiva
                - 🟡 **0.3 a 0.7**: Moderada positiva
                - ⚪ **-0.3 a 0.3**: Fraca
                - 🟡 **-0.7 a -0.3**: Moderada negativa
                - 🔴 **-1.0 a -0.7**: Forte negativa
                
                **Observações:**
                - Valores próximos de **1**: relação direta
                - Valores próximos de **-1**: relação inversa
                - Valores próximos de **0**: sem relação
                """)
            
            st.markdown("---")
            
            st.subheader("📈 Ranking de Correlações com Desempenho")
            
            corr_table = create_correlation_table(
                df_completo,
                'nota_media_5_notas',
                [
                    'escolaridade_pai', 'escolaridade_mae',
                    'ocupacao_pai', 'ocupacao_mae',
                    'faixa_renda', 'pessoas_residencia'
                ],
                method='pearson'
            )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig = plot_correlation_strength(
                    corr_table,
                    title="Força da Correlação com Média Geral"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("##### Tabela de Correlações")
                
                st.dataframe(
                    corr_table.style.format({
                        'Correlação': '{:.3f}',
                        'P-valor': '{:.4f}',
                        'N': '{:.0f}'
                    }).background_gradient(
                        cmap='RdYlGn',
                        subset=['Correlação'],
                        vmin=-1,
                        vmax=1
                    ),
                    use_container_width=True
                )
                
                st.markdown("""
                **Legenda:**
                - **Correlação**: Coeficiente de Pearson
                - **P-valor**: Significância estatística
                - **Significância**: Sim se p < 0.05
                - **N**: Número de observações
                """)
            
            st.markdown("---")
            
            st.success("""
            **💡 Conclusão Principal:** Os dados mostram que existe correlação significativa entre 
            os fatores familiares (escolaridade e ocupação dos pais) e o desempenho no ENEM 2024. 
            A escolaridade dos pais apresenta correlação positiva com as notas, indicando que 
            estudantes cujos pais têm maior nível educacional tendem a ter melhor desempenho.
            """)
