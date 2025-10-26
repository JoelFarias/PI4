"""
Página 1 - Perfil dos Participantes
Análise demográfica e socioeconômica dos participantes do ENEM 2024
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.config import Config
from src.utils.constants import *
from src.database.queries import get_distribuicao_por_campo
from src.data.loader import load_sample_data, get_data_quality_report
from src.visualization.exploratory import plot_categorical_distribution, plot_missing_values


st.set_page_config(
    page_title="Perfil dos Participantes - ENEM 2024",
    page_icon="📊",
    layout=Config.APP_LAYOUT
)

st.title("📊 Perfil dos Participantes")
st.markdown("Análise demográfica e socioeconômica dos participantes do ENEM 2024")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "👥 Perfil Demográfico",
    "💰 Perfil Socioeconômico", 
    "🎓 Perfil Educacional",
    "🔍 Qualidade dos Dados"
])

with tab1:
    st.header("👥 Perfil Demográfico")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuição por Sexo")
        df_sexo = get_distribuicao_por_campo('tp_sexo', limit=10)
        if not df_sexo.empty:
            # Os valores já vêm como 'Masculino' e 'Feminino' do banco
            fig = px.pie(
                df_sexo,
                values='quantidade',
                names='tp_sexo',
                color_discrete_sequence=Config.COLOR_PALETTE,
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Dados não disponíveis")
    
    with col2:
        st.subheader("Distribuição por Cor/Raça")
        df_raca = get_distribuicao_por_campo('tp_cor_raca', limit=10)
        if not df_raca.empty:
            # Os valores já vêm como strings descritivas do banco
            fig = px.bar(
                df_raca,
                x='tp_cor_raca',
                y='quantidade',
                color='tp_cor_raca',
                color_discrete_sequence=Config.COLOR_PALETTE
            )
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Quantidade")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Dados não disponíveis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuição por Faixa Etária")
        df_idade = get_distribuicao_por_campo('tp_faixa_etaria', limit=20)
        if not df_idade.empty:
            # Os valores já vêm como strings descritivas do banco
            fig = px.bar(
                df_idade.head(10),
                x='tp_faixa_etaria',
                y='quantidade',
                color_discrete_sequence=[Config.COLOR_PRIMARY]
            )
            fig.update_layout(xaxis_title="", yaxis_title="Quantidade")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Dados não disponíveis")
    
    with col2:
        st.subheader("Distribuição por Estado Civil")
        df_civil = get_distribuicao_por_campo('tp_estado_civil', limit=10)
        if not df_civil.empty:
            # Os valores já vêm como strings descritivas do banco
            fig = px.pie(
                df_civil,
                values='quantidade',
                names='tp_estado_civil',
                color_discrete_sequence=Config.COLOR_PALETTE
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Dados não disponíveis")
    
    st.subheader("Distribuição Geográfica")
    col1, col2 = st.columns(2)
    
    with col1:
        df_regiao = get_distribuicao_por_campo('regiao_nome_prova', limit=10)
        if not df_regiao.empty:
            fig = px.bar(
                df_regiao,
                x='regiao_nome_prova',
                y='quantidade',
                text='percentual',
                color='regiao_nome_prova',
                color_discrete_sequence=Config.COLOR_PALETTE
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(showlegend=False, xaxis_title="Região", yaxis_title="Quantidade")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        df_uf = get_distribuicao_por_campo('sg_uf_prova', limit=10)
        if not df_uf.empty:
            fig = px.bar(
                df_uf,
                x='sg_uf_prova',
                y='quantidade',
                color_discrete_sequence=[Config.COLOR_SECONDARY]
            )
            fig.update_layout(xaxis_title="UF", yaxis_title="Quantidade")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("💰 Perfil Socioeconômico")
    
    st.subheader("Escolaridade dos Pais")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Escolaridade do Pai (Q001)**")
        df_esc_pai = get_distribuicao_por_campo('q001', limit=10)
        if not df_esc_pai.empty:
            # Os valores já vêm como strings descritivas do banco
            fig = px.bar(
                df_esc_pai,
                x='q001',
                y='quantidade',
                color_discrete_sequence=[Config.COLOR_PRIMARY]
            )
            fig.update_layout(xaxis_title="", yaxis_title="Quantidade")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Dados não disponíveis")
    
    with col2:
        st.markdown("**Escolaridade da Mãe (Q002)**")
        df_esc_mae = get_distribuicao_por_campo('q002', limit=10)
        if not df_esc_mae.empty:
            # Os valores já vêm como strings descritivas do banco
            fig = px.bar(
                df_esc_mae,
                x='q002',
                y='quantidade',
                color_discrete_sequence=[Config.COLOR_SECONDARY]
            )
            fig.update_layout(xaxis_title="", yaxis_title="Quantidade")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Dados não disponíveis")
    
    st.subheader("Ocupação dos Pais")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Ocupação do Pai (Q003)**")
        df_ocup_pai = get_distribuicao_por_campo('q003', limit=10)
        if not df_ocup_pai.empty:
            # Valores já vêm como strings descritivas (truncar para exibição)
            df_ocup_pai['label'] = df_ocup_pai['q003'].str[:50] + '...'
            fig = px.pie(
                df_ocup_pai,
                values='quantidade',
                names='label',
                color_discrete_sequence=Config.COLOR_PALETTE
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Dados não disponíveis")
    
    with col2:
        st.markdown("**Ocupação da Mãe (Q004)**")
        df_ocup_mae = get_distribuicao_por_campo('q004', limit=10)
        if not df_ocup_mae.empty:
            # Valores já vêm como strings descritivas (truncar para exibição)
            df_ocup_mae['label'] = df_ocup_mae['q004'].str[:50] + '...'
            fig = px.pie(
                df_ocup_mae,
                values='quantidade',
                names='label',
                color_discrete_sequence=Config.COLOR_PALETTE
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Dados não disponíveis")
    
    st.subheader("Renda Familiar (Q007)")
    df_renda = get_distribuicao_por_campo('q007', limit=20)
    if not df_renda.empty:
        # Os valores já vêm como strings descritivas do banco
        fig = px.bar(
            df_renda,
            x='q007',
            y='quantidade',
            color='percentual',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(xaxis_title="Faixa de Renda", yaxis_title="Quantidade")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 Dados não disponíveis")

with tab3:
    st.header("🎓 Perfil Educacional")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tipo de Escola (Q023)")
        df_escola = get_distribuicao_por_campo('q023', limit=10)
        if not df_escola.empty:
            # Os valores já vêm como strings descritivas do banco
            fig = px.pie(
                df_escola,
                values='quantidade',
                names='q023',
                color_discrete_sequence=Config.COLOR_PALETTE,
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent')
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar legenda abaixo
            st.caption("**Legenda:**")
            for _, row in df_escola.iterrows():
                st.caption(f"• {row['q023']}: {row['quantidade']:,} ({row['percentual']:.1f}%)")
        else:
            st.info("📊 Dados não disponíveis")
    
    with col2:
        st.subheader("Situação de Conclusão")
        df_conclusao = get_distribuicao_por_campo('tp_st_conclusao', limit=10)
        if not df_conclusao.empty:
            # Os valores já vêm como strings descritivas do banco
            fig = px.bar(
                df_conclusao,
                x='tp_st_conclusao',
                y='quantidade',
                color_discrete_sequence=[Config.COLOR_SUCCESS]
            )
            fig.update_layout(xaxis_title="", yaxis_title="Quantidade")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Dados não disponíveis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ano de Conclusão")
        df_ano = get_distribuicao_por_campo('tp_ano_concluiu', limit=15)
        if not df_ano.empty:
            # Os valores já vêm como strings descritivas do banco
            fig = px.bar(
                df_ano.head(10),
                x='tp_ano_concluiu',
                y='quantidade',
                color_discrete_sequence=[Config.COLOR_WARNING]
            )
            fig.update_layout(xaxis_title="Ano", yaxis_title="Quantidade")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Dados não disponíveis")
    
    with col2:
        st.subheader("Tipo de Ensino")
        df_ensino = get_distribuicao_por_campo('tp_ensino', limit=10)
        if not df_ensino.empty:
            # Os valores já vêm como strings descritivas do banco
            fig = px.bar(
                df_ensino,
                x='tp_ensino',
                y='quantidade',
                color_discrete_sequence=[Config.COLOR_PRIMARY]
            )
            fig.update_layout(xaxis_title="", yaxis_title="Quantidade")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Dados não disponíveis")

with tab4:
    st.header("🔍 Qualidade dos Dados")
    
    st.info("Carregando amostra de 50.000 registros para análise de qualidade...")
    
    with st.spinner("Analisando qualidade dos dados..."):
        df_sample = load_sample_data(n_samples=50000)
        
        if not df_sample.empty:
            quality_report = get_data_quality_report(df_sample)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total de Registros", f"{quality_report['total_rows']:,}".replace(',', '.'))
            with col2:
                st.metric("Total de Colunas", quality_report['total_columns'])
            with col3:
                st.metric("Duplicatas", quality_report['duplicates'])
            with col4:
                st.metric("Uso de Memória", f"{quality_report['memory_usage']:.2f} MB")
            
            st.markdown("---")
            
            st.subheader("Valores Ausentes por Coluna")
            fig = plot_missing_values(quality_report)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Estatísticas de Valores Ausentes")
            missing_df = pd.DataFrame([
                {
                    'Coluna': col,
                    'Valores Ausentes': info['count'],
                    'Percentual': f"{info['percentage']:.2f}%"
                }
                for col, info in quality_report['missing_values'].items()
                if info['percentage'] > 0
            ]).sort_values('Valores Ausentes', ascending=False).head(20)
            
            st.dataframe(missing_df, use_container_width=True)
