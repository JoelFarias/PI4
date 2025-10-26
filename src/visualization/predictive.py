"""
Visualizações para modelos preditivos
Gráficos para análise de regressão, classificação e clustering
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.config import Config


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, title: str = None):
    """
    Scatter plot: valores reais vs preditos
    
    Args:
        y_true: Valores reais
        y_pred: Valores preditos
        title: Título do gráfico
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(
            color=Config.COLOR_PRIMARY,
            opacity=0.6
        ),
        name='Predições'
    ))
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Predição Perfeita'
    ))
    
    fig.update_layout(
        title=title or "Valores Reais vs Preditos",
        xaxis_title="Valores Reais",
        yaxis_title="Valores Preditos",
        height=500,
        showlegend=True
    )
    
    return fig


def plot_residuals(residuals: np.ndarray, y_pred: np.ndarray, title: str = None):
    """
    Plot de resíduos
    
    Args:
        residuals: Resíduos (real - predito)
        y_pred: Valores preditos
        title: Título do gráfico
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(
            color=Config.COLOR_PRIMARY,
            opacity=0.6
        ),
        name='Resíduos'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title=title or "Análise de Resíduos",
        xaxis_title="Valores Preditos",
        yaxis_title="Resíduos",
        height=500
    )
    
    return fig


def plot_residuals_distribution(residuals: np.ndarray, title: str = None):
    """
    Histograma de resíduos
    
    Args:
        residuals: Resíduos
        title: Título do gráfico
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=50,
        marker_color=Config.COLOR_PRIMARY,
        name='Resíduos'
    ))
    
    fig.update_layout(
        title=title or "Distribuição dos Resíduos",
        xaxis_title="Resíduo",
        yaxis_title="Frequência",
        height=400,
        showlegend=False
    )
    
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15, 
                             title: str = None):
    """
    Gráfico de importância de features
    
    Args:
        importance_df: DataFrame com colunas 'feature' e 'importance' ou 'coefficient'
        top_n: Número de features mais importantes a mostrar
        title: Título do gráfico
    """
    if 'importance' in importance_df.columns:
        value_col = 'importance'
        x_label = "Importância"
    else:
        value_col = 'abs_coefficient'
        x_label = "Coeficiente Absoluto"
    
    df_top = importance_df.head(top_n).sort_values(value_col)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_top['feature'],
        x=df_top[value_col],
        orientation='h',
        marker_color=Config.COLOR_PALETTE[:len(df_top)],
        text=[f"{v:.3f}" for v in df_top[value_col]],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=title or f"Top {top_n} Features Mais Importantes",
        xaxis_title=x_label,
        yaxis_title="",
        height=max(400, top_n * 30),
        showlegend=False
    )
    
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: list, title: str = None):
    """
    Matriz de confusão
    
    Args:
        cm: Matriz de confusão
        class_names: Nomes das classes
        title: Título do gráfico
    """
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=cm_normalized,
        x=class_names,
        y=class_names,
        text=cm,
        texttemplate='%{text}',
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title=title or "Matriz de Confusão",
        xaxis_title="Predito",
        yaxis_title="Real",
        height=500
    )
    
    return fig


def plot_roc_curve(roc_data: dict, title: str = None):
    """
    Curva ROC
    
    Args:
        roc_data: dict com 'fpr', 'tpr', 'auc'
        title: Título do gráfico
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=roc_data['fpr'],
        y=roc_data['tpr'],
        mode='lines',
        line=dict(color=Config.COLOR_PRIMARY, width=2),
        name=f"ROC (AUC = {roc_data['auc']:.3f})"
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Chance'
    ))
    
    fig.update_layout(
        title=title or "Curva ROC",
        xaxis_title="Taxa de Falso Positivo",
        yaxis_title="Taxa de Verdadeiro Positivo",
        height=500,
        showlegend=True
    )
    
    return fig


def plot_cluster_scatter_2d(df_pca: pd.DataFrame, title: str = None):
    """
    Scatter plot 2D dos clusters
    
    Args:
        df_pca: DataFrame com PC1, PC2 e cluster
        title: Título do gráfico
    """
    fig = px.scatter(
        df_pca,
        x='PC1',
        y='PC2',
        color='cluster',
        color_discrete_sequence=Config.COLOR_PALETTE,
        labels={'cluster': 'Cluster'},
        title=title or "Visualização dos Clusters (PCA 2D)"
    )
    
    fig.update_layout(height=600)
    
    return fig


def plot_cluster_scatter_3d(df_pca: pd.DataFrame, title: str = None):
    """
    Scatter plot 3D dos clusters
    
    Args:
        df_pca: DataFrame com PC1, PC2, PC3 e cluster
        title: Título do gráfico
    """
    fig = px.scatter_3d(
        df_pca,
        x='PC1',
        y='PC2',
        z='PC3',
        color='cluster',
        color_discrete_sequence=Config.COLOR_PALETTE,
        labels={'cluster': 'Cluster'},
        title=title or "Visualização dos Clusters (PCA 3D)"
    )
    
    fig.update_layout(height=700)
    
    return fig


def plot_elbow_curve(elbow_df: pd.DataFrame, title: str = None):
    """
    Curva do cotovelo para K-Means
    
    Args:
        elbow_df: DataFrame com 'k' e 'inertia'
        title: Título do gráfico
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=elbow_df['k'],
        y=elbow_df['inertia'],
        mode='lines+markers',
        marker=dict(size=10, color=Config.COLOR_PRIMARY),
        line=dict(width=2, color=Config.COLOR_PRIMARY),
        name='Inércia'
    ))
    
    fig.update_layout(
        title=title or "Método do Cotovelo",
        xaxis_title="Número de Clusters (k)",
        yaxis_title="Inércia",
        height=500,
        showlegend=False
    )
    
    return fig


def plot_silhouette_scores(elbow_df: pd.DataFrame, title: str = None):
    """
    Score de silhueta por número de clusters
    
    Args:
        elbow_df: DataFrame com 'k' e 'silhouette'
        title: Título do gráfico
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=elbow_df['k'],
        y=elbow_df['silhouette'],
        mode='lines+markers',
        marker=dict(size=10, color=Config.COLOR_SECONDARY),
        line=dict(width=2, color=Config.COLOR_SECONDARY),
        name='Silhueta'
    ))
    
    fig.update_layout(
        title=title or "Score de Silhueta por K",
        xaxis_title="Número de Clusters (k)",
        yaxis_title="Score de Silhueta",
        height=500,
        showlegend=False
    )
    
    return fig


def plot_cluster_profile_heatmap(profile_df: pd.DataFrame, value_cols: list,
                                   title: str = None):
    """
    Heatmap do perfil de cada cluster
    
    Args:
        profile_df: DataFrame com perfis dos clusters
        value_cols: Colunas de valores para o heatmap
        title: Título do gráfico
    """
    df_heatmap = profile_df[['Cluster'] + value_cols].set_index('Cluster')
    
    fig = px.imshow(
        df_heatmap.T,
        labels=dict(color="Valor"),
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    
    fig.update_layout(
        title=title or "Perfil dos Clusters",
        height=max(400, len(value_cols) * 30)
    )
    
    return fig


def plot_metrics_comparison(metrics_df: pd.DataFrame, metric_cols: list,
                              title: str = None):
    """
    Comparação de métricas entre modelos
    
    Args:
        metrics_df: DataFrame com métricas dos modelos
        metric_cols: Colunas de métricas para comparar
        title: Título do gráfico
    """
    fig = go.Figure()
    
    for i, metric in enumerate(metric_cols):
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_df['Modelo'],
            y=metrics_df[metric],
            text=[f"{v:.3f}" for v in metrics_df[metric]],
            textposition='auto',
            marker_color=Config.COLOR_PALETTE[i % len(Config.COLOR_PALETTE)]
        ))
    
    fig.update_layout(
        title=title or "Comparação de Métricas entre Modelos",
        xaxis_title="Modelo",
        yaxis_title="Score",
        barmode='group',
        height=500,
        showlegend=True
    )
    
    return fig


def plot_learning_curve(train_sizes: np.ndarray, train_scores: np.ndarray,
                         val_scores: np.ndarray, title: str = None):
    """
    Curva de aprendizado
    
    Args:
        train_sizes: Tamanhos de conjunto de treino
        train_scores: Scores de treino
        val_scores: Scores de validação
        title: Título do gráfico
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=train_mean,
        mode='lines+markers',
        name='Treino',
        line=dict(color=Config.COLOR_PRIMARY),
        error_y=dict(type='data', array=train_std, visible=True)
    ))
    
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=val_mean,
        mode='lines+markers',
        name='Validação',
        line=dict(color=Config.COLOR_SECONDARY),
        error_y=dict(type='data', array=val_std, visible=True)
    ))
    
    fig.update_layout(
        title=title or "Curva de Aprendizado",
        xaxis_title="Tamanho do Conjunto de Treino",
        yaxis_title="Score",
        height=500,
        showlegend=True
    )
    
    return fig
