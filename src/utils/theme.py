"""
Sistema de tema minimalista para Streamlit
Aplica CSS que funciona no Streamlit e tema para gráficos Plotly
"""

import streamlit as st


def apply_minimal_theme():
    """
    Aplica tema minimalista ao dashboard.
    Usa apenas seletores CSS que funcionam no Streamlit.
    """
    st.markdown("""
        <style>
        /* ============================================
           TIPOGRAFIA E CORES BASE
           ============================================ */
        
        /* Importar fonte Inter (Google Fonts) */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Aplicar em elementos principais */
        .main .block-container {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        
        /* ============================================
           HEADERS E TÍTULOS
           ============================================ */
        
        .main h1 {
            font-weight: 700;
            color: #1a1a1a;
            letter-spacing: -0.02em;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .main h2 {
            font-weight: 600;
            color: #2d3748;
            font-size: 1.8rem;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        
        .main h3 {
            font-weight: 600;
            color: #2d3748;
            font-size: 1.3rem;
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
        }
        
        /* ============================================
           MÉTRICAS (st.metric)
           ============================================ */
        
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 600;
            color: #1a1a1a;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.875rem;
            font-weight: 500;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        [data-testid="stMetricDelta"] {
            font-size: 0.875rem;
        }
        
        /* ============================================
           SIDEBAR
           ============================================ */
        
        [data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }
        
        [data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }
        
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #1e293b;
        }
        
        /* ============================================
           TABS (st.tabs)
           ============================================ */
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            padding: 0 1rem;
            background-color: transparent;
            border: none;
            border-radius: 0;
            color: #64748b;
            font-weight: 500;
            font-size: 0.95rem;
            border-bottom: 3px solid transparent;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: transparent !important;
            color: #3b82f6;
            border-bottom-color: #3b82f6;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            color: #3b82f6;
            background-color: transparent;
        }
        
        /* ============================================
           BOTÕES (st.button)
           ============================================ */
        
        .stButton > button {
            background-color: #3b82f6;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
            font-size: 0.95rem;
            transition: all 0.2s ease;
        }
        
        .stButton > button:hover {
            background-color: #2563eb;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
            transform: translateY(-1px);
        }
        
        .stButton > button:active {
            transform: translateY(0);
        }
        
        /* ============================================
           INPUTS E SELECTBOX
           ============================================ */
        
        .stSelectbox label,
        .stTextInput label,
        .stNumberInput label,
        .stSlider label {
            font-weight: 500;
            color: #334155;
            font-size: 0.9rem;
        }
        
        /* ============================================
           DATAFRAMES E TABELAS
           ============================================ */
        
        [data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 6px;
        }
        
        /* ============================================
           MENSAGENS (info, success, warning, error)
           ============================================ */
        
        .stAlert {
            border-radius: 6px;
            border: 1px solid;
            padding: 1rem;
        }
        
        /* ============================================
           EXPANDERS (st.expander)
           ============================================ */
        
        .streamlit-expanderHeader {
            font-weight: 500;
            color: #1e293b;
            background-color: #f8fafc;
            border-radius: 6px;
            border: 1px solid #e2e8f0;
        }
        
        .streamlit-expanderHeader:hover {
            border-color: #cbd5e1;
        }
        
        /* ============================================
           DIVISORES (hr)
           ============================================ */
        
        hr {
            margin: 2rem 0;
            border: none;
            border-top: 1px solid #e2e8f0;
        }
        
        /* ============================================
           SPINNER (st.spinner)
           ============================================ */
        
        .stSpinner > div {
            border-top-color: #3b82f6 !important;
        }
        
        /* ============================================
           CHARTS - CONTAINER
           ============================================ */
        
        [data-testid="stPlotlyChart"] {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            background: white;
        }
        
        /* ============================================
           CUSTOM CLASSES (para HTML customizado)
           ============================================ */
        
        .info-box {
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 1rem 1.25rem;
            border-radius: 4px;
            margin: 1.5rem 0;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        
        .info-box h4 {
            margin-top: 0;
            margin-bottom: 0.5rem;
            font-size: 1rem;
            font-weight: 600;
            color: #1e40af;
        }
        
        .info-box p {
            margin-bottom: 0.5rem;
            color: #1e40af;
        }
        
        .warning-box {
            background: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 1rem 1.25rem;
            border-radius: 4px;
            margin: 1.5rem 0;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        
        .warning-box h4 {
            margin-top: 0;
            margin-bottom: 0.5rem;
            font-size: 1rem;
            font-weight: 600;
            color: #92400e;
        }
        
        .warning-box p {
            margin-bottom: 0.5rem;
            color: #92400e;
        }
        
        .success-box {
            background: #f0fdf4;
            border-left: 4px solid #10b981;
            padding: 1rem 1.25rem;
            border-radius: 4px;
            margin: 1.5rem 0;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        
        .success-box h4 {
            margin-top: 0;
            margin-bottom: 0.5rem;
            font-size: 1rem;
            font-weight: 600;
            color: #065f46;
        }
        
        /* ============================================
           RESPONSIVIDADE
           ============================================ */
        
        @media (max-width: 768px) {
            .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            
            .main h1 {
                font-size: 2rem;
            }
            
            [data-testid="stMetricValue"] {
                font-size: 1.5rem;
            }
        }
        
        </style>
    """, unsafe_allow_html=True)


def get_plotly_theme():
    """
    Retorna configuração de tema minimalista para gráficos Plotly.
    Aplicar em cada gráfico com fig.update_layout(**get_plotly_theme())
    
    Returns:
        dict: Configurações de layout para Plotly
    """
    return {
        'font': {
            'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            'size': 13,
            'color': '#1e293b'
        },
        'plot_bgcolor': '#ffffff',
        'paper_bgcolor': '#ffffff',
        'colorway': ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'],
        'margin': {'t': 60, 'r': 30, 'b': 50, 'l': 60},
        'xaxis': {
            'gridcolor': '#f1f5f9',
            'linecolor': '#e2e8f0',
            'showgrid': True,
            'zeroline': False,
            'title': {'font': {'size': 12, 'color': '#64748b'}},
            'tickmode': 'auto',
            'automargin': True
        },
        'yaxis': {
            'gridcolor': '#f1f5f9',
            'linecolor': '#e2e8f0',
            'showgrid': True,
            'zeroline': False,
            'title': {'font': {'size': 12, 'color': '#64748b'}},
            'automargin': True
        },
        'legend': {
            'bgcolor': 'rgba(255,255,255,0.9)',
            'bordercolor': '#e2e8f0',
            'borderwidth': 1,
            'font': {'size': 11}
        },
        'hovermode': 'closest',
        'hoverlabel': {
            'bgcolor': '#1e293b',
            'font': {'family': 'Inter, sans-serif', 'size': 12, 'color': 'white'},
            'bordercolor': '#1e293b'
        }
    }


def section_header(title: str, subtitle: str = None):
    """
    Cria um cabeçalho de seção minimalista.
    
    Args:
        title: Título da seção
        subtitle: Subtítulo opcional
    """
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)
    st.markdown("")


def wrap_text(text, max_length=20):
    """
    Quebra texto em múltiplas linhas para labels de eixos.
    
    Args:
        text: Texto a ser quebrado
        max_length: Comprimento máximo por linha
    
    Returns:
        str: Texto com quebras de linha (<br>)
    """
    if not isinstance(text, str) or len(text) <= max_length:
        return text
    
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= max_length:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '<br>'.join(lines)


def apply_wrapped_labels(fig, axis='x', max_length=20):
    """
    Aplica wrap automático aos labels de um eixo do gráfico Plotly.
    
    Args:
        fig: Figura Plotly
        axis: 'x' ou 'y'
        max_length: Comprimento máximo por linha
    
    Returns:
        fig: Figura modificada
    """
    import plotly.graph_objects as go
    
    if axis == 'x':
        # Obter dados do eixo X
        if hasattr(fig, 'data') and len(fig.data) > 0:
            if hasattr(fig.data[0], 'x'):
                original_labels = fig.data[0].x
                if original_labels is not None:
                    wrapped_labels = [wrap_text(str(label), max_length) for label in original_labels]
                    fig.update_xaxes(ticktext=wrapped_labels, tickvals=list(range(len(wrapped_labels))))
    
    return fig


def info_card(title: str, content: str, card_type: str = "info"):
    """
    Cria um card informativo minimalista.
    
    Args:
        title: Título do card
        content: Conteúdo (pode conter HTML)
        card_type: Tipo (info, warning, success)
    """
    box_class = f"{card_type}-box"
    st.markdown(f"""
    <div class="{box_class}">
        <h4>{title}</h4>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)
