"""
Dashboard ENEM 2024 - Análise de Fatores Familiares
Página Inicial

Autor: Joel C.
Instituição: IESB
Disciplina: PI-4
Data: Outubro 2025
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Importar módulos do projeto
from src.utils.config import Config
from src.utils.constants import (
    TEXTO_CONTEXTUALIZACAO,
    NOMES_PROVAS,
    REGIOES,
)
from src.utils.theme import apply_minimal_theme, get_plotly_theme, wrap_text
from src.database.connection import test_database_connection
from src.database.queries import (
    get_notas_estatisticas,
    get_distribuicao_por_campo,
)


# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================

st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon="📊",
    layout=Config.APP_LAYOUT,
    initial_sidebar_state="expanded",
)

# Aplicar tema minimalista
apply_minimal_theme()


# ==============================================================================
# CABEÇALHO PRINCIPAL
# ==============================================================================

st.title("Dashboard ENEM 2024")
st.markdown("Análise de Correlação entre Fatores Familiares e Desempenho Acadêmico em Nível Municipal")
st.caption("Sistema de análise ecológica baseado em dados agregados por município")

# AVISO IMPORTANTE SOBRE NÍVEL DE ANÁLISE
st.markdown("""
<div class="warning-box">
    <h4>⚠️ IMPORTANTE: Análise em Nível Municipal (Análise Ecológica)</h4>
    <p>Este dashboard realiza <strong>análise ecológica</strong> - os dados são agregados por <strong>MUNICÍPIO</strong>, não por participante individual.</p>
    <p><strong>Por quê?</strong> As tabelas de dados socioeconômicos e resultados do ENEM estão separadas e só podem ser relacionadas através do município de prova.</p>
    <p><strong>O que isso significa?</strong></p>
    <ul>
        <li>Cada "observação" representa um município (~5.570 municípios)</li>
        <li>Features são <strong>percentuais/médias municipais</strong> (ex: "% de pais com ensino superior no município")</li>
        <li>Target é a <strong>média de desempenho do município</strong></li>
        <li>Correlações encontradas são entre características dos municípios, NÃO entre indivíduos</li>
    </ul>
    <p><strong>⚠️ Falácia Ecológica:</strong> Correlações no nível municipal NÃO implicam que o mesmo padrão exista no nível individual. 
    Exemplo: "Municípios com mais pais universitários têm melhor desempenho" ≠ "Alunos com pais universitários têm melhor desempenho".</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ==============================================================================
# CONTEXTUALIZAÇÃO DO PROBLEMA
# ==============================================================================

st.markdown("## 🎯 Sobre o Projeto")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(TEXTO_CONTEXTUALIZACAO)
    
    st.markdown("### 🔍 Questão de Pesquisa")
    st.markdown("""
    > **"Qual é a correlação entre o desempenho dos alunos no ENEM 2024 
    > e o nível de escolaridade e ocupação dos seus pais?"**
    """)
    
    st.markdown("### 🎯 Objetivos")
    st.markdown("""
    - ✅ Analisar a distribuição de desempenho no ENEM 2024
    - ✅ Identificar correlações entre fatores socioeconômicos e notas
    - ✅ Desenvolver modelos preditivos de desempenho
    - ✅ Visualizar padrões geográficos de desempenho
    - ✅ Gerar insights acionáveis para políticas educacionais
    """)

with col2:
    st.markdown("### 📊 Fonte de Dados")
    st.info("""
    **Banco de Dados:** PostgreSQL  
    **Servidor:** IESB  
    **Registros:** 4.3M participantes  
    **Período:** ENEM 2024  
    **Atualização:** Tempo real
    """)
    
    st.markdown("### 🔗 Variáveis Principais")
    st.success("""
    **Dependentes:**
    - Notas (CN, CH, LC, MT, Redação)
    
    **Independentes:**
    - Escolaridade dos pais
    - Ocupação dos pais  
    - Renda familiar
    - Fatores demográficos
    """)

st.markdown("---")


# ==============================================================================
# MÉTRICAS GERAIS (KPIs)
# ==============================================================================

st.markdown("## 📈 Visão Geral do ENEM 2024")

# Carregar estatísticas
with st.spinner("Carregando estatísticas gerais..."):
    try:
        stats = get_notas_estatisticas()
        
        if not stats.empty:
            stats_dict = stats.iloc[0].to_dict()
            
            # Tratar valores None para evitar erros de formatação
            for key in stats_dict:
                if stats_dict[key] is None:
                    stats_dict[key] = 0.0
            
            # Primeira linha de métricas - Total e Médias Principais
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "👥 Total de Participantes",
                    f"{int(stats_dict['total_participantes']):,}".replace(",", "."),
                    help="Número total de participantes com notas válidas"
                )
            
            with col2:
                st.metric(
                    "📊 Média Geral",
                    f"{stats_dict['media_geral']:.1f}",
                    help="Média das 5 notas (CN, CH, LC, MT, Redação)"
                )
            
            with col3:
                st.metric(
                    "🔬 Ciências da Natureza",
                    f"{stats_dict['cn_media']:.1f}",
                    delta=f"{stats_dict['cn_media'] - stats_dict['media_geral']:.1f}",
                    help="Média em Ciências da Natureza"
                )
            
            with col4:
                st.metric(
                    "📚 Ciências Humanas",
                    f"{stats_dict['ch_media']:.1f}",
                    delta=f"{stats_dict['ch_media'] - stats_dict['media_geral']:.1f}",
                    help="Média em Ciências Humanas"
                )
            
            with col5:
                st.metric(
                    "📝 Linguagens e Códigos",
                    f"{stats_dict['lc_media']:.1f}",
                    delta=f"{stats_dict['lc_media'] - stats_dict['media_geral']:.1f}",
                    help="Média em Linguagens e Códigos"
                )
            
            # Segunda linha de métricas - Outras áreas
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "➗ Matemática",
                    f"{stats_dict['mt_media']:.1f}",
                    delta=f"{stats_dict['mt_media'] - stats_dict['media_geral']:.1f}",
                    help="Média em Matemática"
                )
            
            with col2:
                st.metric(
                    "✍️ Redação",
                    f"{stats_dict['red_media']:.1f}",
                    delta=f"{stats_dict['red_media'] - stats_dict['media_geral']:.1f}",
                    help="Média em Redação"
                )
            
            with col3:
                st.metric(
                    "📊 Mediana Geral",
                    f"{stats_dict['media_mediana']:.1f}",
                    help="Mediana das notas gerais"
                )
            
            with col4:
                st.metric(
                    "📏 Desvio Padrão",
                    f"{stats_dict['media_desvio']:.1f}",
                    help="Dispersão das notas"
                )
            
            with col5:
                st.metric(
                    "🎯 Amplitude",
                    f"{stats_dict['media_max'] - stats_dict['media_min']:.1f}",
                    help="Diferença entre maior e menor nota"
                )
            
            st.markdown("---")
            
            # Comparação visual de médias
            st.markdown("### 📊 Comparação de Médias por Área")
            
            # Preparar dados para o gráfico
            areas = ['CN', 'CH', 'LC', 'MT', 'Redação']
            medias = [
                float(stats_dict['cn_media']),
                float(stats_dict['ch_media']),
                float(stats_dict['lc_media']),
                float(stats_dict['mt_media']),
                float(stats_dict['red_media'])
            ]
            
            # Criar gráfico de barras
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=areas,
                y=medias,
                text=[f"{m:.1f}" for m in medias],
                textposition='auto',
                marker_color=Config.COLOR_PALETTE[:5],
                hovertemplate='<b>%{x}</b><br>Média: %{y:.2f}<extra></extra>'
            ))
            
            # Adicionar linha de referência (média geral)
            fig.add_hline(
                y=stats_dict['media_geral'],
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"Média Geral: {stats_dict['media_geral']:.1f}",
                annotation_position="right"
            )
            
            theme = get_plotly_theme()
            fig.update_layout(
                **{k: v for k, v in theme.items() if k not in ['title', 'margin', 'xaxis', 'yaxis', 'hovermode']},
                title="Média por Área de Conhecimento",
                xaxis=dict(
                    title="Área de Conhecimento",
                    tickangle=0,
                    tickfont=dict(size=12)
                ),
                yaxis=dict(
                    title="Pontuação Média",
                    range=[0, max(medias) * 1.15],
                    tickformat=".1f"
                ),
                height=450,
                showlegend=False,
                hovermode='x unified',
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("⚠️ Não foi possível carregar as estatísticas. Verifique a conexão com o banco de dados.")
    
    except Exception as e:
        st.error(f"❌ Erro ao carregar estatísticas: {str(e)}")
        if Config.DEBUG_MODE:
            st.exception(e)

st.markdown("---")


# ==============================================================================
# DISTRIBUIÇÕES BÁSICAS
# ==============================================================================

st.markdown("## 📊 Distribuições Demográficas")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👥 Distribuição por Sexo")
    with st.spinner("Carregando dados..."):
        try:
            df_sexo = get_distribuicao_por_campo('tp_sexo')
            
            if not df_sexo.empty and len(df_sexo) > 0:
                # Os valores já vêm como 'Feminino' e 'Masculino' do banco
                # Não é necessário mapeamento
                
                fig = px.pie(
                    df_sexo,
                    values='quantidade',
                    names='tp_sexo',
                    color_discrete_sequence=Config.COLOR_PALETTE,
                    hole=0.4
                )
                
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    textfont_size=14,
                    pull=[0.05, 0.05],
                    hovertemplate='<b>%{label}</b><br>' +
                                  'Quantidade: %{value:,.0f}<br>' +
                                  'Percentual: %{percent:.1f}%<extra></extra>'
                )
                
                theme = get_plotly_theme()
                fig.update_layout(
                    **{k: v for k, v in theme.items() if k not in ['margin', 'legend']},
                    height=400,
                    margin=dict(t=30, b=0, l=0, r=0),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.15,
                        xanchor="center",
                        x=0.5
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 Dados não disponíveis")
        except Exception as e:
            st.error(f"❌ Erro: {str(e)}")
            if Config.DEBUG_MODE:
                st.exception(e)

with col2:
    st.markdown("### 🌍 Distribuição por Região")
    with st.spinner("Carregando dados..."):
        try:
            df_regiao = get_distribuicao_por_campo('regiao_nome_prova')
            
            if not df_regiao.empty:
                fig = px.bar(
                    df_regiao,
                    x='regiao_nome_prova',
                    y='quantidade',
                    text='percentual',
                    color='regiao_nome_prova',
                    color_discrete_sequence=Config.COLOR_PALETTE
                )
                
                fig.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside',
                    textfont=dict(size=12, color='#1f2937'),
                    hovertemplate='<b>%{x}</b><br>' +
                                  'Participantes: %{y:,.0f}<br>' +
                                  'Percentual: %{text:.1f}%<extra></extra>'
                )
                
                theme = get_plotly_theme()
                fig.update_layout(
                    **{k: v for k, v in theme.items() if k not in ['xaxis', 'yaxis', 'margin', 'legend']},
                    showlegend=False,
                    xaxis=dict(
                        title="Região",
                        tickangle=0,
                        tickfont=dict(size=11)
                    ),
                    yaxis=dict(
                        title="Número de Participantes",
                        tickformat=",",
                        separatethousands=True
                    ),
                    height=400,
                    margin=dict(t=30, b=80, l=60, r=30)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Dados não disponíveis")
        except Exception as e:
            st.error(f"Erro: {str(e)}")

st.markdown("---")


# ==============================================================================
# CALL TO ACTION
# ==============================================================================

st.markdown("## 🚀 Explore o Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📊 Análise Exploratória
    
    Explore o perfil dos participantes, distribuições de notas
    e características demográficas.
    
    👉 **Acesse:** Páginas 1 e 2 na sidebar
    """)

with col2:
    st.markdown("""
    ### 🔗 Análise de Correlação
    
    Descubra como fatores familiares se correlacionam
    com o desempenho acadêmico.
    
    👉 **Acesse:** Página 3 na sidebar
    """)

with col3:
    st.markdown("""
    ### 🤖 Modelos Preditivos
    
    Veja predições baseadas em machine learning
    e identifique padrões ocultos.
    
    👉 **Acesse:** Páginas 4, 5 e 6 na sidebar
    """)

st.markdown("---")


# ==============================================================================
# RODAPÉ
# ==============================================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📚 Fonte de Dados:**  
    INEP - Microdados ENEM 2024  
    FIRJAN - Índice de Desenvolvimento Municipal
    """)

with col2:
    st.markdown("""
    **🏫 Instituição:**  
    IESB - Centro Universitário  
    Disciplina: PI-4
    """)

with col3:
    st.markdown(f"""
    **📅 Atualização:**  
    {datetime.now().strftime('%d/%m/%Y %H:%M')}  
    Versão: 1.0.0
    """)

st.markdown("---")
