"""
Funções de visualização para análise de correlação
Gráficos especializados para mostrar relações entre variáveis
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr
import streamlit as st

from src.utils.config import Config


def plot_correlation_heatmap(df: pd.DataFrame, columns: list = None, 
                              method: str = 'pearson', title: str = None):
    """
    Cria heatmap de correlação entre variáveis
    
    Args:
        df: DataFrame com os dados
        columns: Lista de colunas para calcular correlação
        method: 'pearson' ou 'spearman'
        title: Título do gráfico
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_num = df[columns].dropna()
    
    if method == 'pearson':
        corr_matrix = df_num.corr()
    else:
        corr_matrix = df_num.corr(method='spearman')
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlação"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        text_auto='.2f',
        aspect='auto'
    )
    
    fig.update_layout(
        title=title or f"Matriz de Correlação ({method.capitalize()})",
        xaxis_title="",
        yaxis_title="",
        height=600
    )
    
    return fig


def plot_scatter_with_regression(df: pd.DataFrame, x_col: str, y_col: str, 
                                   x_label: str = None, y_label: str = None,
                                   color_col: str = None, title: str = None):
    """
    Gráfico de dispersão com linha de regressão
    
    Args:
        df: DataFrame com os dados
        x_col: Coluna do eixo X
        y_col: Coluna do eixo Y
        x_label: Label do eixo X
        y_label: Label do eixo Y
        color_col: Coluna para colorir pontos
        title: Título do gráfico
    """
    df_clean = df[[x_col, y_col]].dropna()
    
    if len(df_clean) == 0:
        st.warning("Não há dados suficientes para criar o gráfico")
        return None
    
    fig = px.scatter(
        df_clean,
        x=x_col,
        y=y_col,
        color=color_col if color_col and color_col in df.columns else None,
        trendline="ols",
        opacity=0.6
    )
    
    pearson_corr, p_value = pearsonr(df_clean[x_col], df_clean[y_col])
    
    fig.update_layout(
        title=title or f"{x_label or x_col} vs {y_label or y_col}",
        xaxis_title=x_label or x_col,
        yaxis_title=y_label or y_col,
        height=500,
        annotations=[
            dict(
                x=0.05,
                y=0.95,
                xref='paper',
                yref='paper',
                text=f'Correlação: {pearson_corr:.3f}<br>p-valor: {p_value:.4f}',
                showarrow=False,
                bgcolor='white',
                bordercolor='black',
                borderwidth=1
            )
        ]
    )
    
    return fig


def plot_grouped_boxplot(df: pd.DataFrame, category_col: str, value_col: str,
                          category_labels: dict = None, value_label: str = None,
                          title: str = None):
    """
    Box plot agrupado por categoria
    
    Args:
        df: DataFrame com os dados
        category_col: Coluna categórica
        value_col: Coluna de valores
        category_labels: Dicionário de mapeamento de categorias
        value_label: Label do eixo Y
        title: Título do gráfico
    """
    df_clean = df[[category_col, value_col]].dropna()
    
    if category_labels:
        df_clean[category_col] = df_clean[category_col].map(category_labels)
    
    category_order = df_clean.groupby(category_col)[value_col].median().sort_values(ascending=False).index.tolist()
    
    fig = px.box(
        df_clean,
        x=category_col,
        y=value_col,
        color=category_col,
        category_orders={category_col: category_order}
    )
    
    fig.update_layout(
        title=title or f"{value_label or value_col} por {category_col}",
        xaxis_title=category_col,
        yaxis_title=value_label or value_col,
        showlegend=False,
        height=500
    )
    
    return fig


def plot_grouped_bar_with_error(df: pd.DataFrame, category_col: str, value_col: str,
                                  category_labels: dict = None, value_label: str = None,
                                  title: str = None):
    """
    Gráfico de barras com média e barra de erro
    
    Args:
        df: DataFrame com os dados
        category_col: Coluna categórica
        value_col: Coluna de valores
        category_labels: Dicionário de mapeamento
        value_label: Label do eixo Y
        title: Título do gráfico
    """
    df_clean = df[[category_col, value_col]].dropna()
    
    if category_labels:
        df_clean[category_col] = df_clean[category_col].map(category_labels)
    
    stats = df_clean.groupby(category_col)[value_col].agg(['mean', 'std', 'count']).reset_index()
    stats = stats.sort_values('mean', ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=stats[category_col],
        y=stats['mean'],
        error_y=dict(type='data', array=stats['std']),
        text=[f"{m:.1f}" for m in stats['mean']],
        textposition='auto',
        marker_color=Config.COLOR_PALETTE[:len(stats)]
    ))
    
    fig.update_layout(
        title=title or f"Média de {value_label or value_col} por {category_col}",
        xaxis_title=category_col,
        yaxis_title=f"Média de {value_label or value_col}",
        height=500,
        showlegend=False
    )
    
    return fig, stats


def plot_violin_comparison(df: pd.DataFrame, category_col: str, value_col: str,
                            category_labels: dict = None, value_label: str = None,
                            title: str = None):
    """
    Violin plot para comparação de distribuições
    
    Args:
        df: DataFrame com os dados
        category_col: Coluna categórica
        value_col: Coluna de valores
        category_labels: Dicionário de mapeamento
        value_label: Label do eixo Y
        title: Título do gráfico
    """
    df_clean = df[[category_col, value_col]].dropna()
    
    if category_labels:
        df_clean[category_col] = df_clean[category_col].map(category_labels)
    
    fig = px.violin(
        df_clean,
        x=category_col,
        y=value_col,
        color=category_col,
        box=True,
        points='outliers'
    )
    
    fig.update_layout(
        title=title or f"Distribuição de {value_label or value_col} por {category_col}",
        xaxis_title=category_col,
        yaxis_title=value_label or value_col,
        showlegend=False,
        height=500
    )
    
    return fig


def create_correlation_table(df: pd.DataFrame, target_col: str, feature_cols: list,
                               method: str = 'pearson'):
    """
    Cria tabela com correlações e p-valores
    
    Args:
        df: DataFrame com os dados
        target_col: Coluna alvo
        feature_cols: Lista de colunas para correlacionar
        method: 'pearson' ou 'spearman'
    
    Returns:
        DataFrame com correlações ordenadas
    """
    correlations = []
    
    for col in feature_cols:
        df_clean = df[[col, target_col]].dropna()
        
        if len(df_clean) < 3:
            continue
        
        if method == 'pearson':
            corr, p_val = pearsonr(df_clean[col], df_clean[target_col])
        else:
            corr, p_val = spearmanr(df_clean[col], df_clean[target_col])
        
        correlations.append({
            'Variável': col,
            'Correlação': corr,
            'P-valor': p_val,
            'Significância': 'Sim' if p_val < 0.05 else 'Não',
            'N': len(df_clean)
        })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Correlação', key=abs, ascending=False)
    
    return corr_df


def plot_correlation_strength(corr_df: pd.DataFrame, title: str = None):
    """
    Gráfico de barras horizontais com força da correlação
    
    Args:
        corr_df: DataFrame com colunas 'Variável' e 'Correlação'
        title: Título do gráfico
    """
    corr_df_sorted = corr_df.sort_values('Correlação', key=abs)
    
    colors = ['red' if c < 0 else 'green' for c in corr_df_sorted['Correlação']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=corr_df_sorted['Variável'],
        x=corr_df_sorted['Correlação'],
        orientation='h',
        marker_color=colors,
        text=[f"{c:.3f}" for c in corr_df_sorted['Correlação']],
        textposition='auto'
    ))
    
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    
    fig.update_layout(
        title=title or "Força da Correlação",
        xaxis_title="Coeficiente de Correlação",
        yaxis_title="",
        height=max(400, len(corr_df_sorted) * 30),
        showlegend=False
    )
    
    return fig


def plot_faceted_scatter(df: pd.DataFrame, x_col: str, y_col: str, facet_col: str,
                          x_label: str = None, y_label: str = None, title: str = None):
    """
    Scatter plots separados por categoria (facets)
    
    Args:
        df: DataFrame com os dados
        x_col: Coluna do eixo X
        y_col: Coluna do eixo Y
        facet_col: Coluna para criar facets
        x_label: Label do eixo X
        y_label: Label do eixo Y
        title: Título do gráfico
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        facet_col=facet_col,
        facet_col_wrap=3,
        trendline="ols",
        opacity=0.5
    )
    
    fig.update_layout(
        title=title or f"{y_label or y_col} vs {x_label or x_col}",
        height=600
    )
    
    fig.update_xaxes(title_text=x_label or x_col)
    fig.update_yaxes(title_text=y_label or y_col)
    
    return fig
