"""
Módulo de visualizações exploratórias.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Optional, List, Dict

from ..utils.config import Config
from ..utils.constants import NOMES_PROVAS, SIGLAS_PROVAS


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    title: str,
    bins: int = 50,
    color: Optional[str] = None
) -> go.Figure:
    """Cria histograma de distribuição."""
    
    fig = px.histogram(
        df,
        x=column,
        nbins=bins,
        title=title,
        color_discrete_sequence=[color or Config.COLOR_PRIMARY],
        marginal="box"
    )
    
    fig.update_layout(
        height=Config.DEFAULT_CHART_HEIGHT,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig


def plot_boxplot(
    df: pd.DataFrame,
    columns: List[str],
    title: str
) -> go.Figure:
    """Cria boxplot para múltiplas variáveis."""
    
    fig = go.Figure()
    
    for i, col in enumerate(columns):
        fig.add_trace(go.Box(
            y=df[col].dropna(),
            name=SIGLAS_PROVAS.get(col, col),
            marker_color=Config.COLOR_PALETTE[i % len(Config.COLOR_PALETTE)]
        ))
    
    fig.update_layout(
        title=title,
        height=Config.DEFAULT_CHART_HEIGHT,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def plot_statistics_table(stats: pd.DataFrame, columns: List[str]) -> go.Figure:
    """Cria tabela de estatísticas descritivas."""
    
    stats_data = []
    for col in columns:
        stats_data.append({
            'Prova': SIGLAS_PROVAS.get(col, col),
            'Média': f"{stats[f'{col.split("_")[1]}_media'].iloc[0]:.1f}",
            'Mediana': f"{stats[f'{col.split("_")[1]}_mediana'].iloc[0]:.1f}",
            'Desvio': f"{stats[f'{col.split("_")[1]}_desvio'].iloc[0]:.1f}",
            'Mín': f"{stats[f'{col.split("_")[1]}_min'].iloc[0]:.1f}",
            'Máx': f"{stats[f'{col.split("_")[1]}_max'].iloc[0]:.1f}"
        })
    
    df_table = pd.DataFrame(stats_data)
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df_table.columns),
            fill_color=Config.COLOR_PRIMARY,
            font=dict(color='white', size=12),
            align='center'
        ),
        cells=dict(
            values=[df_table[col] for col in df_table.columns],
            fill_color='lavender',
            align='center'
        )
    )])
    
    fig.update_layout(height=300)
    
    return fig


def plot_categorical_distribution(
    df: pd.DataFrame,
    column: str,
    title: str,
    top_n: int = 10,
    chart_type: str = 'bar'
) -> go.Figure:
    """Plota distribuição de variável categórica."""
    
    value_counts = df[column].value_counts().head(top_n)
    
    if chart_type == 'bar':
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=title,
            labels={'x': column, 'y': 'Quantidade'},
            color_discrete_sequence=[Config.COLOR_PRIMARY]
        )
    elif chart_type == 'pie':
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=title,
            color_discrete_sequence=Config.COLOR_PALETTE
        )
    
    fig.update_layout(height=Config.DEFAULT_CHART_HEIGHT)
    
    return fig


def plot_grouped_statistics(
    df: pd.DataFrame,
    group_by: str,
    value_column: str,
    title: str,
    mapping: Optional[Dict] = None
) -> go.Figure:
    """Plota estatísticas agrupadas."""
    
    grouped = df.groupby(group_by)[value_column].agg(['mean', 'median', 'std', 'count']).reset_index()
    
    if mapping:
        grouped[group_by] = grouped[group_by].map(mapping)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Média e Mediana', 'Desvio Padrão e Contagem'),
        specs=[[{"secondary_y": False}, {"secondary_y": True}]]
    )
    
    fig.add_trace(
        go.Bar(name='Média', x=grouped[group_by], y=grouped['mean'], 
               marker_color=Config.COLOR_PRIMARY),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(name='Mediana', x=grouped[group_by], y=grouped['median'],
                   mode='lines+markers', marker_color=Config.COLOR_SECONDARY),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(name='Desvio Padrão', x=grouped[group_by], y=grouped['std'],
               marker_color=Config.COLOR_WARNING),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(name='Contagem', x=grouped[group_by], y=grouped['count'],
                   mode='lines+markers', marker_color=Config.COLOR_SUCCESS),
        row=1, col=2, secondary_y=True
    )
    
    fig.update_layout(
        title_text=title,
        height=Config.DEFAULT_CHART_HEIGHT,
        showlegend=True
    )
    
    return fig


def plot_missing_values(quality_report: Dict) -> go.Figure:
    """Plota gráfico de valores ausentes."""
    
    missing_data = quality_report['missing_values']
    
    columns = []
    percentages = []
    
    for col, info in missing_data.items():
        if info['percentage'] > 0:
            columns.append(col)
            percentages.append(info['percentage'])
    
    df_missing = pd.DataFrame({
        'Coluna': columns,
        'Percentual': percentages
    }).sort_values('Percentual', ascending=True).tail(20)
    
    fig = px.bar(
        df_missing,
        x='Percentual',
        y='Coluna',
        orientation='h',
        title='Top 20 Colunas com Valores Ausentes',
        color='Percentual',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(height=600)
    
    return fig
