"""
Página 5 - Análise Geoespacial
Análise de desempenho por município, UF e região com integração do IFDM
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.utils.config import Config
from src.utils.constants import REGIOES
from src.database.queries import (
    get_media_por_municipio,
    get_media_por_uf,
    get_media_por_regiao
)
from src.data.loader import load_ifdm_data
from src.visualization.maps import (
    plot_top_municipalities,
    plot_region_comparison,
    plot_ifdm_correlation,
    plot_heatmap_by_region_subject,
    create_performance_summary_table
)


st.set_page_config(
    page_title="Análise Geográfica - ENEM 2024",
    page_icon="🗺️",
    layout=Config.APP_LAYOUT
)

st.title("🗺️ Análise Geoespacial")
st.markdown("Análise de desempenho por localização geográfica e desenvolvimento municipal")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🌎 Por Região",
    "📍 Por Estado (UF)",
    "🏘️ Por Município",
    "📊 IFDM x Desempenho"
])

with tab1:
    st.header("🌎 Análise por Região")
    
    with st.spinner("Carregando dados por região..."):
        df_regiao = get_media_por_regiao()
        
        if not df_regiao.empty:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Ranking de Regiões")
                
                st.dataframe(
                    df_regiao.set_index('regiao').style.format({
                        'total_participantes': '{:,.0f}',
                        'cn_media': '{:.1f}',
                        'ch_media': '{:.1f}',
                        'lc_media': '{:.1f}',
                        'mt_media': '{:.1f}',
                        'red_media': '{:.1f}',
                        'media_geral': '{:.1f}'
                    }).background_gradient(cmap='RdYlGn', subset=['media_geral']),
                    use_container_width=True
                )
            
            with col2:
                st.subheader("📈 Comparação Visual")
                
                fig = plot_region_comparison(
                    df_regiao.set_index('regiao'),
                    value_col='media_geral'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🎯 Desempenho por Disciplina")
            
            with st.spinner("Gerando heatmap..."):
                df_sample = df_regiao.copy()
                df_sample['nota_cn_ciencias_da_natureza'] = df_sample['cn_media']
                df_sample['nota_ch_ciencias_humanas'] = df_sample['ch_media']
                df_sample['nota_lc_linguagens_e_codigos'] = df_sample['lc_media']
                df_sample['nota_mt_matematica'] = df_sample['mt_media']
                df_sample['nota_redacao'] = df_sample['red_media']
                
                fig = plot_heatmap_by_region_subject(df_sample, region_col='regiao')
                st.plotly_chart(fig, use_container_width=True)
            
            melhor_regiao = df_regiao.iloc[0]['regiao']
            pior_regiao = df_regiao.iloc[-1]['regiao']
            diferenca = df_regiao.iloc[0]['media_geral'] - df_regiao.iloc[-1]['media_geral']
            
            st.info(f"""
            **💡 Insight Regional:** A região **{melhor_regiao}** apresenta o melhor desempenho médio 
            ({df_regiao.iloc[0]['media_geral']:.1f} pontos), enquanto **{pior_regiao}** possui a menor 
            média ({df_regiao.iloc[-1]['media_geral']:.1f} pontos). A diferença entre elas é de 
            **{diferenca:.1f} pontos**.
            """)

with tab2:
    st.header("📍 Análise por Estado (UF)")
    
    with st.spinner("Carregando dados por UF..."):
        df_uf = get_media_por_uf()
        
        if not df_uf.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Ranking Completo de Estados")
                
                st.dataframe(
                    df_uf.style.format({
                        'total_participantes': '{:,.0f}',
                        'cn_media': '{:.1f}',
                        'ch_media': '{:.1f}',
                        'lc_media': '{:.1f}',
                        'mt_media': '{:.1f}',
                        'red_media': '{:.1f}',
                        'media_geral': '{:.1f}'
                    }).background_gradient(cmap='RdYlGn', subset=['media_geral']),
                    use_container_width=True,
                    height=600
                )
            
            with col2:
                st.subheader("🏆 Top 10 Estados")
                
                top_10 = df_uf.head(10)
                
                fig = plot_region_comparison(
                    top_10.set_index('uf'),
                    value_col='media_geral',
                    title="Top 10 Estados - Média Geral"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.subheader("⚠️ Bottom 10 Estados")
                
                bottom_10 = df_uf.tail(10).sort_values('media_geral')
                
                fig = plot_region_comparison(
                    bottom_10.set_index('uf'),
                    value_col='media_geral',
                    title="Bottom 10 Estados - Média Geral"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📈 Análise por Disciplina")
            
            disciplina_selecionada = st.selectbox(
                "Selecione a disciplina:",
                options=[
                    ('cn_media', 'Ciências da Natureza'),
                    ('ch_media', 'Ciências Humanas'),
                    ('lc_media', 'Linguagens e Códigos'),
                    ('mt_media', 'Matemática'),
                    ('red_media', 'Redação')
                ],
                format_func=lambda x: x[1],
                key='disciplina_uf'
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                top_disc = df_uf.nlargest(15, disciplina_selecionada[0])
                
                fig = plot_region_comparison(
                    top_disc.set_index('uf'),
                    value_col=disciplina_selecionada[0],
                    title=f"Top 15 - {disciplina_selecionada[1]}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                bottom_disc = df_uf.nsmallest(15, disciplina_selecionada[0]).sort_values(disciplina_selecionada[0])
                
                fig = plot_region_comparison(
                    bottom_disc.set_index('uf'),
                    value_col=disciplina_selecionada[0],
                    title=f"Bottom 15 - {disciplina_selecionada[1]}"
                )
                st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🏘️ Análise por Município")
    
    st.markdown("""
    Análise dos municípios com maior e menor desempenho médio no ENEM 2024.
    **Critério:** Municípios com no mínimo 30 participantes.
    """)
    
    top_n_mun = st.slider(
        "Número de municípios a exibir",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
        key='top_n_mun'
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Melhores Municípios")
        
        with st.spinner("Carregando melhores municípios..."):
            df_top_mun = get_media_por_municipio(top_n=top_n_mun, order='DESC')
            
            if not df_top_mun.empty:
                fig = plot_top_municipalities(
                    df_top_mun,
                    value_col='media_geral',
                    top_n=top_n_mun,
                    ascending=False,
                    title=f"Top {top_n_mun} Melhores Municípios"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(
                    df_top_mun[['municipio', 'uf', 'media_geral', 'quantidade_participantes']].style.format({
                        'media_geral': '{:.1f}',
                        'quantidade_participantes': '{:,.0f}'
                    }).background_gradient(cmap='Greens', subset=['media_geral']),
                    use_container_width=True
                )
    
    with col2:
        st.subheader("⚠️ Piores Municípios")
        
        with st.spinner("Carregando piores municípios..."):
            df_bottom_mun = get_media_por_municipio(top_n=top_n_mun, order='ASC')
            
            if not df_bottom_mun.empty:
                fig = plot_top_municipalities(
                    df_bottom_mun,
                    value_col='media_geral',
                    top_n=top_n_mun,
                    ascending=True,
                    title=f"Top {top_n_mun} Piores Municípios"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(
                    df_bottom_mun[['municipio', 'uf', 'media_geral', 'quantidade_participantes']].style.format({
                        'media_geral': '{:.1f}',
                        'quantidade_participantes': '{:,.0f}'
                    }).background_gradient(cmap='Reds_r', subset=['media_geral']),
                    use_container_width=True
                )
    
    if not df_top_mun.empty and not df_bottom_mun.empty:
        melhor_cidade = df_top_mun.iloc[0]['municipio']
        melhor_uf = df_top_mun.iloc[0]['uf']
        melhor_nota = df_top_mun.iloc[0]['media_geral']
        
        pior_cidade = df_bottom_mun.iloc[0]['municipio']
        pior_uf = df_bottom_mun.iloc[0]['uf']
        pior_nota = df_bottom_mun.iloc[0]['media_geral']
        
        diferenca_cidades = melhor_nota - pior_nota
        
        st.warning(f"""
        **💡 Disparidade Municipal:** O município de **{melhor_cidade}-{melhor_uf}** apresenta média de 
        **{melhor_nota:.1f} pontos**, enquanto **{pior_cidade}-{pior_uf}** tem apenas **{pior_nota:.1f} pontos**. 
        A diferença é de **{diferenca_cidades:.1f} pontos**, evidenciando grande desigualdade regional no país.
        """)

with tab4:
    st.header("📊 IFDM x Desempenho ENEM")
    
    st.markdown("""
    **IFDM** - Índice Firjan de Desenvolvimento Municipal
    
    Análise da correlação entre o desenvolvimento municipal (IFDM) e o desempenho no ENEM 2024.
    """)
    
    with st.spinner("Carregando dados do IFDM..."):
        df_ifdm = load_ifdm_data()
        
        if df_ifdm is not None and not df_ifdm.empty:
            st.success(f"✅ Dados IFDM carregados: {len(df_ifdm):,} municípios")
            
            ano_ifdm = st.selectbox(
                "Selecione o ano do IFDM:",
                options=sorted([col for col in df_ifdm.columns if 'IFDM' in col and any(str(y) in col for y in range(2019, 2024))], reverse=True),
                key='ano_ifdm'
            )
            
            if ano_ifdm:
                with st.spinner("Carregando desempenho por município..."):
                    df_municipios = get_media_por_municipio(top_n=5570, order='DESC')
                
                if not df_municipios.empty:
                    df_merged = df_municipios.merge(
                        df_ifdm[['Nome do Município', ano_ifdm]],
                        left_on='municipio',
                        right_on='Nome do Município',
                        how='inner'
                    )
                    
                    df_merged = df_merged.dropna(subset=[ano_ifdm, 'media_geral'])
                    
                    if not df_merged.empty:
                        st.info(f"📌 Cruzamento realizado: {len(df_merged):,} municípios com dados completos")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.subheader("📈 Correlação IFDM x Desempenho")
                            
                            fig = plot_ifdm_correlation(
                                df_merged,
                                ifdm_col=ano_ifdm,
                                performance_col='media_geral',
                                title=f"{ano_ifdm} x Média Geral ENEM 2024"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.subheader("📊 Estatísticas")
                            
                            from scipy.stats import pearsonr, spearmanr
                            
                            corr_pearson, p_pearson = pearsonr(
                                df_merged[ano_ifdm],
                                df_merged['media_geral']
                            )
                            corr_spearman, p_spearman = spearmanr(
                                df_merged[ano_ifdm],
                                df_merged['media_geral']
                            )
                            
                            st.metric("Correlação de Pearson", f"{corr_pearson:.3f}")
                            st.metric("P-valor (Pearson)", f"{p_pearson:.4f}")
                            
                            st.metric("Correlação de Spearman", f"{corr_spearman:.3f}")
                            st.metric("P-valor (Spearman)", f"{p_spearman:.4f}")
                            
                            if p_pearson < 0.05:
                                st.success("✅ Correlação estatisticamente significativa")
                            else:
                                st.warning("⚠️ Correlação não significativa (p > 0.05)")
                        
                        st.markdown("---")
                        st.subheader("🎯 Municípios de Destaque")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("##### 🏆 Alto IFDM + Alto Desempenho")
                            
                            ifdm_q75 = df_merged[ano_ifdm].quantile(0.75)
                            nota_q75 = df_merged['media_geral'].quantile(0.75)
                            
                            df_destaque = df_merged[
                                (df_merged[ano_ifdm] >= ifdm_q75) &
                                (df_merged['media_geral'] >= nota_q75)
                            ].nlargest(10, 'media_geral')
                            
                            st.dataframe(
                                df_destaque[['municipio', 'uf', ano_ifdm, 'media_geral']].style.format({
                                    ano_ifdm: '{:.4f}',
                                    'media_geral': '{:.1f}'
                                }).background_gradient(cmap='Greens'),
                                use_container_width=True
                            )
                        
                        with col2:
                            st.markdown("##### ⚠️ Baixo IFDM + Baixo Desempenho")
                            
                            ifdm_q25 = df_merged[ano_ifdm].quantile(0.25)
                            nota_q25 = df_merged['media_geral'].quantile(0.25)
                            
                            df_atencao = df_merged[
                                (df_merged[ano_ifdm] <= ifdm_q25) &
                                (df_merged['media_geral'] <= nota_q25)
                            ].nsmallest(10, 'media_geral')
                            
                            st.dataframe(
                                df_atencao[['municipio', 'uf', ano_ifdm, 'media_geral']].style.format({
                                    ano_ifdm: '{:.4f}',
                                    'media_geral': '{:.1f}'
                                }).background_gradient(cmap='Reds_r'),
                                use_container_width=True
                            )
                        
                        st.success(f"""
                        **💡 Conclusão IFDM:** Existe correlação **{'positiva' if corr_pearson > 0 else 'negativa'}** 
                        de **{abs(corr_pearson):.3f}** entre o IFDM e o desempenho no ENEM. 
                        Isso indica que municípios com maior desenvolvimento tendem a ter estudantes 
                        com melhor desempenho na prova.
                        """)
                    else:
                        st.error("Não foi possível cruzar os dados IFDM com municípios do ENEM")
                else:
                    st.error("Erro ao carregar dados de municípios")
        else:
            st.warning("""
            ⚠️ **Dados IFDM não disponíveis**
            
            Para habilitar esta análise:
            1. Verifique se o arquivo `ipeadata[29-09-2025-03-52].csv` está na pasta `data/`
            2. O arquivo deve conter dados do IFDM (Índice Firjan) por município
            """)

st.markdown("---")
st.info("""
**📌 Observações:**
- Dados agregados por município consideram apenas localidades com mínimo de 30 participantes
- IFDM varia de 0 a 1 (quanto mais próximo de 1, melhor o desenvolvimento)
- Análise geográfica permite identificar disparidades regionais e oportunidades de políticas públicas
""")
