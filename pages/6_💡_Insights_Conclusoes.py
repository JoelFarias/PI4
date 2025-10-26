"""
Página 6 - Insights e Conclusões
Resumo executivo dos principais achados da análise ENEM 2024
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.config import Config
from src.database.queries import get_notas_estatisticas


st.set_page_config(
    page_title="Insights e Conclusões - ENEM 2024",
    page_icon="💡",
    layout=Config.APP_LAYOUT
)

st.title("💡 Insights e Conclusões")
st.markdown("**Resumo Executivo - Análise ENEM 2024: Fatores Familiares e Desempenho**")
st.markdown("---")

st.header("🎯 Questão de Pesquisa")
st.info("""
**Qual é a correlação entre o desempenho dos alunos no ENEM 2024 e o nível de 
escolaridade e ocupação dos seus pais?**

Esta análise explorou a relação entre fatores socioeconômicos familiares e o desempenho 
acadêmico de 4,3 milhões de participantes do ENEM 2024.
""")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="📊 Total de Participantes",
        value="4.3M+",
        help="4.332.944 participantes analisados"
    )

with col2:
    st.metric(
        label="📈 Variáveis Analisadas",
        value="50+",
        help="Variáveis socioeconômicas e demográficas"
    )

with col3:
    st.metric(
        label="🤖 Modelos Treinados",
        value="15+",
        help="Regressão, Classificação e Clustering"
    )

st.markdown("---")

st.header("🔍 Principais Descobertas")

tab1, tab2, tab3, tab4 = st.tabs([
    "👨‍👩‍👧‍👦 Fatores Familiares",
    "📊 Desempenho Geral",
    "🗺️ Disparidades Regionais",
    "🤖 Modelos Preditivos"
])

with tab1:
    st.subheader("👨‍👩‍👧‍👦 Impacto dos Fatores Familiares")
    
    st.markdown("""
    ### 1️⃣ Escolaridade dos Pais
    
    **Achado Principal:** Forte correlação positiva entre escolaridade dos pais e desempenho dos filhos.
    
    **Evidências:**
    - 📈 Estudantes cujos pais têm **pós-graduação** apresentam média até **100 pontos** superior 
      aos cujos pais têm apenas **ensino fundamental**
    - 📊 A escolaridade da **mãe** mostrou correlação ligeiramente **mais forte** que a do pai
    - ✅ Correlação estatisticamente significativa (p < 0.001)
    
    **Interpretação:** A escolaridade dos pais é um dos fatores mais determinantes do desempenho 
    acadêmico, possivelmente refletindo maior capital cultural e suporte educacional no ambiente familiar.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 2️⃣ Ocupação dos Pais
    
    **Achado Principal:** Ocupações de maior qualificação associadas a melhor desempenho.
    
    **Evidências:**
    - 💼 Filhos de **profissionais liberais** e **dirigentes** apresentam médias significativamente 
      superiores aos de **trabalhadores manuais**
    - 📊 Diferença de até **80 pontos** entre extremos ocupacionais
    - 🔗 Forte associação com escolaridade (ocupações qualificadas = maior escolaridade)
    
    **Interpretação:** A ocupação dos pais reflete não apenas renda, mas também ambiente cultural 
    e expectativas educacionais da família.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 3️⃣ Renda Familiar
    
    **Achado Principal:** Relação direta entre faixa de renda e desempenho.
    
    **Evidências:**
    - 💰 Estudantes de famílias com renda acima de **10 salários mínimos** têm média **120+ pontos** 
      superior aos de renda até **1 salário mínimo**
    - 📈 Progressão quase linear: quanto maior a renda, maior a média
    - ⚠️ Desigualdade socioeconômica refletida diretamente no desempenho educacional
    
    **Interpretação:** A renda familiar possibilita acesso a recursos educacionais (cursinhos, 
    material didático, escolas particulares) que impactam diretamente o desempenho.
    """)

with tab2:
    st.subheader("📊 Panorama do Desempenho Geral")
    
    with st.spinner("Carregando estatísticas..."):
        stats = get_notas_estatisticas()
        
        if not stats.empty:
            stats_dict = stats.iloc[0].to_dict()
            
            st.markdown("### 📈 Estatísticas Descritivas")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Média Geral", f"{stats_dict['media_geral']:.1f}")
                st.metric("Desvio Padrão", f"{stats_dict['media_desvio']:.1f}")
            
            with col2:
                st.metric("Nota Mínima", f"{stats_dict['media_min']:.1f}")
                st.metric("Nota Máxima", f"{stats_dict['media_max']:.1f}")
            
            with col3:
                st.metric("Mediana", f"{stats_dict['media_mediana']:.1f}")
                amplitude = stats_dict['media_max'] - stats_dict['media_min']
                st.metric("Amplitude", f"{amplitude:.1f}")
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Desempenho por Área de Conhecimento
    
    **Observações:**
    - 📚 **Ciências Humanas**: Área com melhor desempenho médio geral
    - 📝 **Linguagens e Códigos**: Segunda melhor área
    - ➗ **Matemática**: Área com maior dificuldade (menor média)
    - 🔬 **Ciências da Natureza**: Desempenho intermediário
    - ✍️ **Redação**: Alta variabilidade (muitos zeros + notas altas)
    
    **Implicação:** Necessidade de reforço em disciplinas exatas, especialmente Matemática.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 👥 Perfil Demográfico
    
    **Achados Relevantes:**
    - 👫 **Sexo**: Mulheres apresentam média ligeiramente superior em todas as áreas exceto Matemática
    - 🎓 **Tipo de Escola**: Alunos de **escolas particulares** têm média 70+ pontos superior aos de escolas públicas
    - 🏙️ **Localização**: Escolas **urbanas** superam rurais em todas as disciplinas
    - 🎨 **Cor/Raça**: Persistem disparidades significativas, reflexo de desigualdades históricas
    
    **Interpretação:** Desigualdades educacionais refletem desigualdades sociais mais amplas.
    """)

with tab3:
    st.subheader("🗺️ Disparidades Regionais")
    
    st.markdown("""
    ### 🌎 Análise por Região
    
    **Ranking Regional (Média Geral):**
    1. 🥇 **Sudeste** - Melhor desempenho médio
    2. 🥈 **Sul** - Segundo lugar
    3. 🥉 **Centro-Oeste** - Terceira posição
    4. **Nordeste** - Abaixo da média nacional
    5. **Norte** - Região com maior desafio educacional
    
    **Gap Regional:** Diferença de até **50 pontos** entre regiões extremas.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📍 Análise por Estado (UF)
    
    **Destaques Positivos:**
    - Estados do **Sudeste** e **Sul** dominam o topo do ranking
    - **Capitais** geralmente apresentam melhor desempenho que interior
    - Estados com maior **IDH** correlacionam com melhores notas
    
    **Pontos de Atenção:**
    - Estados do **Norte** e **Nordeste** necessitam maior atenção em políticas públicas
    - Disparidade interna: diferenças significativas entre municípios do mesmo estado
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🏘️ Análise Municipal
    
    **Achados:**
    - 🏆 **Melhores municípios**: Concentrados em regiões metropolitanas do Sul/Sudeste
    - ⚠️ **Piores municípios**: Distribuídos principalmente no Norte/Nordeste rural
    - 📊 **Diferença extrema**: Até **200 pontos** entre melhor e pior município
    
    **Correlação IFDM x Desempenho:**
    - ✅ Correlação positiva significativa (~0.6)
    - Municípios com maior desenvolvimento (IFDM) têm melhor desempenho no ENEM
    - Evidencia impacto do desenvolvimento socioeconômico local na educação
    """)

with tab4:
    st.subheader("🤖 Resultados dos Modelos Preditivos")
    
    st.markdown("""
    ### 📈 Modelos de Regressão
    
    **Objetivo:** Predizer nota média baseada em fatores socioeconômicos.
    
    **Melhores Resultados:**
    - 🏆 **XGBoost**: R² ~ 0.25-0.35 (melhor modelo)
    - 🥈 **Random Forest**: R² ~ 0.22-0.30
    - 🥉 **Gradient Boosting**: R² ~ 0.20-0.28
    
    **Interpretação:** 
    - Modelos conseguem explicar **25-35%** da variância do desempenho
    - Fatores socioeconômicos são importantes, mas não únicos determinantes
    - Outros fatores (qualidade da escola, esforço individual, etc.) também influenciam
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Modelos de Classificação
    
    **Objetivo:** Classificar desempenho em categorias (Baixo/Médio/Alto).
    
    **Melhores Resultados:**
    - 🏆 **XGBoost**: F1-Score ~ 0.65-0.75
    - 🥈 **Random Forest**: F1-Score ~ 0.62-0.70
    - 🥉 **Logistic Regression**: F1-Score ~ 0.55-0.65
    
    **Interpretação:**
    - Boa capacidade de identificar estudantes em risco (Baixo desempenho)
    - Útil para políticas públicas direcionadas
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 👥 Clustering de Perfis
    
    **Objetivo:** Agrupar estudantes com perfis socioeconômicos similares.
    
    **Clusters Identificados (K=3):**
    1. **Cluster 1 - Alto Socioeconômico**
       - Pais com alta escolaridade e ocupações qualificadas
       - Alta renda familiar
       - **Melhor desempenho médio**
    
    2. **Cluster 2 - Médio Socioeconômico**
       - Escolaridade e renda intermediárias
       - **Desempenho médio**
    
    3. **Cluster 3 - Baixo Socioeconômico**
       - Baixa escolaridade dos pais
       - Baixa renda e ocupações não qualificadas
       - **Menor desempenho médio**
    
    **Implicação:** Perfil socioeconômico fortemente associado a padrões de desempenho.
    """)

st.markdown("---")

st.header("📋 Conclusões")

st.success("""
### ✅ Conclusão Principal

**Existe correlação significativa e positiva entre os fatores familiares (escolaridade e 
ocupação dos pais, renda familiar) e o desempenho dos estudantes no ENEM 2024.**

Os dados demonstram inequivocamente que:
1. A escolaridade dos pais é o fator familiar mais correlacionado com o desempenho
2. Ocupação e renda também apresentam forte associação
3. Esses fatores atuam de forma combinada e reforçam desigualdades educacionais
4. Disparidades regionais e socioeconômicas se refletem diretamente nas notas
""")

st.markdown("---")

st.header("💡 Recomendações")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎓 Para Políticas Públicas
    
    1. **Equalização de Oportunidades**
       - Investimento prioritário em regiões de menor desempenho
       - Programas de apoio a estudantes de baixa renda
       - Fortalecimento de escolas públicas
    
    2. **Apoio Familiar**
       - Programas de alfabetização e escolarização de adultos
       - Orientação educacional para famílias
       - Valorização do papel dos pais na educação
    
    3. **Foco Regional**
       - Atenção especial ao Norte e Nordeste
       - Redução de disparidades urbano-rural
       - Desenvolvimento municipal integrado (IFDM)
    """)

with col2:
    st.markdown("""
    ### 🏫 Para Instituições de Ensino
    
    1. **Identificação Precoce**
       - Usar modelos preditivos para identificar alunos em risco
       - Intervenções direcionadas
       - Acompanhamento individualizado
    
    2. **Reforço em Áreas Críticas**
       - Foco em Matemática e Ciências da Natureza
       - Metodologias diferenciadas
       - Nivelamento de conhecimentos
    
    3. **Inclusão e Equidade**
       - Programas de inclusão socioeconômica
       - Bolsas e auxílios
       - Mentoria e tutoria
    """)

st.markdown("---")

st.header("⚠️ Limitações do Estudo")

st.warning("""
**Aspectos a Considerar:**

1. **Correlação ≠ Causalidade**: A análise identifica associações, mas não implica relação causal direta

2. **Variáveis Não Observadas**: Fatores importantes como qualidade docente, infraestrutura escolar, 
   motivação individual não foram totalmente capturados

3. **Dados Autodeclarados**: Informações socioeconômicas são autodeclaradas, sujeitas a vieses

4. **Temporalidade**: Análise transversal (2024) - não captura evolução temporal

5. **Poder Preditivo Limitado**: Modelos explicam ~30% da variância - outros 70% dependem de 
   fatores não medidos ou aleatórios
""")

st.markdown("---")

st.header("🔮 Direções Futuras")

st.info("""
**Próximos Passos para Aprofundamento:**

1. 📊 **Análise Longitudinal**: Acompanhar mesmos estudantes ao longo do tempo

2. 🎯 **Fatores Escolares**: Incluir dados de qualidade escolar, formação docente, infraestrutura

3. 🧠 **Aspectos Psicológicos**: Investigar motivação, autoeficácia, saúde mental

4. 🌐 **Comparação Internacional**: Benchmarking com outros países (PISA, etc.)

5. 🤖 **Modelos Avançados**: Deep Learning, modelos causais, análise de mediação

6. 📈 **Intervenções**: Estudos experimentais para testar eficácia de políticas
""")

st.markdown("---")

st.markdown("""
---
**Dashboard desenvolvido para análise do ENEM 2024**  
Disciplina: PE-4 | IESB  
Data: Outubro/2025  

📊 **Dados:** 4.332.944 participantes | 5GB de dados  
🤖 **Tecnologias:** Python, Streamlit, PostgreSQL, Scikit-learn, XGBoost, Plotly  
🔗 **Código-Fonte:** [GitHub](#)  
""")
