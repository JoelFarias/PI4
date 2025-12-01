"""
Visualizações geoespaciais para análise por município
Mapas interativos com desempenho e dados do IFDM
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.config import Config
from src.utils.theme import get_plotly_theme


def plot_choropleth_map(df: pd.DataFrame, location_col: str, value_col: str,
                         locationmode: str = 'geojson-id', geojson = None,
                         title: str = None, color_scale: str = 'Blues'):
    """
    Mapa coroplético (colorido por valores)
    
    Args:
        df: DataFrame com dados
        location_col: Coluna com códigos de localização
        value_col: Coluna com valores para colorir
        locationmode: Modo de localização
        geojson: Dados GeoJSON (opcional)
        title: Título do mapa
        color_scale: Escala de cores
    """
    fig = px.choropleth(
        df,
        locations=location_col,
        color=value_col,
        locationmode=locationmode,
        color_continuous_scale=color_scale,
        title=title or f"Mapa: {value_col}",
        hover_data=df.columns.tolist()
    )
    
    fig.update_geos(
        fitbounds="locations",
        visible=False
    )
    
    fig.update_layout(
        height=600,
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    
    return fig


def plot_scatter_map(df: pd.DataFrame, lat_col: str, lon_col: str, 
                      size_col: str = None, color_col: str = None,
                      hover_data: list = None, title: str = None):
    """
    Mapa de pontos com coordenadas lat/lon
    
    Args:
        df: DataFrame com dados
        lat_col: Coluna de latitude
        lon_col: Coluna de longitude
        size_col: Coluna para tamanho dos pontos
        color_col: Coluna para cor dos pontos
        hover_data: Colunas para mostrar no hover
        title: Título do mapa
    """
    fig = px.scatter_geo(
        df,
        lat=lat_col,
        lon=lon_col,
        size=size_col,
        color=color_col,
        hover_data=hover_data,
        title=title or "Mapa de Localidades"
    )
    
    fig.update_geos(
        scope='south america',
        showcountries=True,
        showcoastlines=True,
        showland=True,
        fitbounds="locations"
    )
    
    fig.update_layout(height=600)
    
    return fig


def plot_state_performance_map(df_estado: pd.DataFrame, value_col: str = 'media_geral',
                                 title: str = None):
    """
    Mapa do Brasil por estados
    
    Args:
        df_estado: DataFrame com dados por UF
        value_col: Coluna com valores (ex: media_geral)
        title: Título do mapa
    """
    uf_to_state = {
        'AC': 'Acre', 'AL': 'Alagoas', 'AP': 'Amapá', 'AM': 'Amazonas',
        'BA': 'Bahia', 'CE': 'Ceará', 'DF': 'Distrito Federal', 'ES': 'Espírito Santo',
        'GO': 'Goiás', 'MA': 'Maranhão', 'MT': 'Mato Grosso', 'MS': 'Mato Grosso do Sul',
        'MG': 'Minas Gerais', 'PA': 'Pará', 'PB': 'Paraíba', 'PR': 'Paraná',
        'PE': 'Pernambuco', 'PI': 'Piauí', 'RJ': 'Rio de Janeiro', 'RN': 'Rio Grande do Norte',
        'RS': 'Rio Grande do Sul', 'RO': 'Rondônia', 'RR': 'Roraima', 'SC': 'Santa Catarina',
        'SP': 'São Paulo', 'SE': 'Sergipe', 'TO': 'Tocantins'
    }
    
    if 'uf' in df_estado.columns:
        df_estado['state_name'] = df_estado['uf'].map(uf_to_state)
    
    fig = px.choropleth(
        df_estado,
        locations='uf' if 'uf' in df_estado.columns else df_estado.index,
        locationmode="USA-states",
        color=value_col,
        scope="south america",
        color_continuous_scale='RdYlGn',
        title=title or f"Desempenho por Estado - {value_col}"
    )
    
    fig.update_geos(
        center={"lat": -14, "lon": -55},
        projection_scale=4,
        visible=False
    )
    
    fig.update_layout(
        height=600,
        geo=dict(
            showframe=False,
            showcoastlines=False,
        )
    )
    
    return fig


def plot_top_municipalities(df: pd.DataFrame, value_col: str = 'media_geral',
                             top_n: int = 20, ascending: bool = False, 
                             title: str = None):
    """
    Gráfico de barras dos top N municípios
    
    Args:
        df: DataFrame com dados por município
        value_col: Coluna de valor
        top_n: Número de municípios a mostrar
        ascending: Se True, mostra os piores
        title: Título do gráfico
    """
    df_sorted = df.sort_values(value_col, ascending=ascending).head(top_n)
    
    if 'municipio_nome' not in df_sorted.columns:
        df_sorted['municipio_nome'] = df_sorted.index
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['municipio_nome'],
        x=df_sorted[value_col],
        orientation='h',
        marker_color=Config.COLOR_PALETTE[:len(df_sorted)],
        text=[f"{v:.1f}" for v in df_sorted[value_col]],
        textposition='auto'
    ))
    
    ranking_type = "Piores" if ascending else "Melhores"
    
    fig.update_layout(
        title=title or f"Top {top_n} {ranking_type} Municípios - {value_col}",
        xaxis_title=value_col,
        yaxis_title="",
        height=max(400, top_n * 25),
        showlegend=False
    )
    
    return fig


def plot_region_comparison(df_regiao: pd.DataFrame, value_col: str = 'media_geral',
                             title: str = None):
    """
    Comparação de desempenho por região
    
    Args:
        df_regiao: DataFrame com dados por região
        value_col: Coluna de valor
        title: Título do gráfico
    """
    df_sorted = df_regiao.sort_values(value_col, ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_sorted.index if 'regiao' not in df_sorted.columns else df_sorted['regiao'],
        y=df_sorted[value_col],
        marker_color=Config.COLOR_PALETTE[:len(df_sorted)],
        text=[f"{v:.1f}" for v in df_sorted[value_col]],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=title or f"Desempenho por Região - {value_col}",
        xaxis_title="Região",
        yaxis_title=value_col,
        height=500,
        showlegend=False
    )
    
    return fig


def plot_ifdm_correlation(df: pd.DataFrame, ifdm_col: str, 
                           performance_col: str = 'media_geral',
                           title: str = None):
    """
    Scatter plot: IFDM vs Desempenho ENEM
    
    Args:
        df: DataFrame com IFDM e desempenho
        ifdm_col: Coluna do IFDM
        performance_col: Coluna de desempenho
        title: Título do gráfico
    """
    fig = px.scatter(
        df,
        x=ifdm_col,
        y=performance_col,
        trendline="ols",
        opacity=0.6,
        hover_data=df.columns.tolist(),
        color_discrete_sequence=[Config.COLOR_PRIMARY]
    )
    
    from scipy.stats import pearsonr
    if len(df) > 2:
        corr, p_val = pearsonr(df[ifdm_col].dropna(), df[performance_col].dropna())
        
        fig.add_annotation(
            x=0.05,
            y=0.95,
            xref='paper',
            yref='paper',
            text=f'Correlação: {corr:.3f}<br>P-valor: {p_val:.4f}',
            showarrow=False,
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        )
    
    fig.update_layout(
        title=title or f"{ifdm_col} vs {performance_col}",
        xaxis_title=ifdm_col,
        yaxis_title=performance_col,
        height=500
    )
    
    return fig


def plot_bubble_map(df: pd.DataFrame, x_col: str, y_col: str, size_col: str,
                     color_col: str = None, hover_name: str = None,
                     title: str = None):
    """
    Gráfico de bolhas para análise multivariada
    
    Args:
        df: DataFrame com dados
        x_col: Coluna eixo X
        y_col: Coluna eixo Y
        size_col: Coluna para tamanho das bolhas
        color_col: Coluna para cor
        hover_name: Coluna para nome no hover
        title: Título do gráfico
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=size_col,
        color=color_col,
        hover_name=hover_name,
        size_max=60,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        title=title or f"{y_col} vs {x_col}",
        height=600
    )
    
    return fig


def create_performance_summary_table(df: pd.DataFrame, group_col: str,
                                       value_cols: list):
    """
    Cria tabela resumo de desempenho por grupo
    
    Args:
        df: DataFrame com dados
        group_col: Coluna para agrupar
        value_cols: Colunas de valores para agregar
    
    Returns:
        DataFrame com estatísticas por grupo
    """
    summary = df.groupby(group_col)[value_cols].agg(['mean', 'std', 'count'])
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    return summary.sort_values(f'{value_cols[0]}_mean', ascending=False)


def plot_heatmap_by_region_subject(df: pd.DataFrame, region_col: str = 'regiao',
                                     subject_cols: list = None, title: str = None):
    """
    Heatmap de desempenho por região e disciplina
    
    Args:
        df: DataFrame com dados
        region_col: Coluna de região
        subject_cols: Colunas das disciplinas
        title: Título do gráfico
    """
    if subject_cols is None:
        subject_cols = [
            'nota_cn_ciencias_da_natureza',
            'nota_ch_ciencias_humanas',
            'nota_lc_linguagens_e_codigos',
            'nota_mt_matematica',
            'nota_redacao'
        ]
    
    pivot = df.groupby(region_col)[subject_cols].mean()
    
    pivot.columns = ['CN', 'CH', 'LC', 'MT', 'Redação']
    
    fig = px.imshow(
        pivot.T,
        labels=dict(color="Média"),
        color_continuous_scale='RdYlGn',
        aspect='auto',
        text_auto='.1f'
    )
    
    fig.update_layout(
        title=title or "Desempenho Médio por Região e Disciplina",
        xaxis_title="Região",
        yaxis_title="Disciplina",
        height=500
    )
    
    return fig
