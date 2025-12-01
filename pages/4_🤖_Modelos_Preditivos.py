"""
Página 4 - Modelos Preditivos
Interface para treinamento e avaliação de modelos de Machine Learning
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.utils.config import Config
from src.utils.constants import *
from src.utils.theme import apply_minimal_theme, get_plotly_theme
from src.data.loader import load_municipio_data
from src.database.queries import get_participantes_sample
from src.models import (
    RegressionModel, prepare_regression_data, compare_models,
    ClassificationModel, prepare_classification_data, compare_classifiers,
    ClusteringModel, find_optimal_k, compare_clustering_methods
)
from src.visualization.predictive import (
    plot_actual_vs_predicted, plot_residuals, plot_residuals_distribution,
    plot_feature_importance, plot_confusion_matrix, plot_roc_curve,
    plot_cluster_scatter_2d, plot_cluster_scatter_3d, plot_elbow_curve,
    plot_silhouette_scores, plot_cluster_profile_heatmap, plot_metrics_comparison
)


st.set_page_config(
    page_title="Modelos Preditivos - ENEM 2024",
    page_icon="🤖",
    layout=Config.APP_LAYOUT
)

apply_minimal_theme()

st.title("🤖 Modelos Preditivos")
st.markdown("""
Aplicação de Machine Learning para análise e predição de desempenho no ENEM 2024.
""")
st.markdown("---")

# Sub-abas dentro de cada tipo de modelo
tab1, tab2, tab3 = st.tabs([
    "📈 Regressão",
    "🎯 Classificação",
    "👥 Clustering"
])

with tab1:
    st.header("📈 Modelos de Regressão")
    st.markdown("Predição da **nota média** baseada em fatores socioeconômicos")
    
    st.info("""📊 **Análise Ecológica (Nível Municipal)**  
    - Cada observação representa um **município** (não participantes individuais)
    - Features são **percentuais/médias municipais** (ex: % pais com ensino superior)
    - Target é **média municipal** das notas
    - ⚠️ Correlações municipais ≠ relações individuais (falácia ecológica)
    """)
    
    # Configurações na sidebar
    st.sidebar.subheader("⚙️ Configurações - Regressão")
    
    sample_size_reg = st.sidebar.slider(
        "Número de municípios",
        min_value=100,
        max_value=5500,
        value=3000,
        step=500,
        key='sample_reg',
        help="Total de municípios disponíveis: ~5.570"
    )
    
    test_size_reg = st.sidebar.slider(
        "% Teste",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
        key='test_size_reg'
    )
    
    # Criar sub-abas para modelos individuais
    model_tabs = st.tabs([
        "📊 Linear",
        "📊 Random Forest",
        "📊 XGBoost",
        "📉 Comparação"
    ])
    
    # Preparar dados uma única vez (usado por todos os modelos)
    @st.cache_data(ttl=3600)
    def get_regression_data(sample_size, test_size):
        """Carrega dados MUNICIPAIS para regressão (análise ecológica)"""
        df_municipios = load_municipio_data(min_participantes=30)
        
        if df_municipios.empty:
            return None, None, None, None
        
        # Features municipais (percentuais/médias)
        feature_cols = [
            'perc_pai_ensino_superior',
            'perc_mae_ensino_superior',
            'perc_renda_alta',
            'perc_renda_baixa',
            'media_pessoas_residencia',
            'perc_escola_publica'
        ]
        
        # Amostrar municípios (não participantes individuais)
        if len(df_municipios) > sample_size:
            df_sample = df_municipios.sample(n=min(sample_size, len(df_municipios)), random_state=42)
        else:
            df_sample = df_municipios
        
        return prepare_regression_data(
            df_sample,
            target_col='media_geral',
            feature_cols=feature_cols,
            test_size=test_size
        )
    
    X_train, X_test, y_train, y_test = get_regression_data(sample_size_reg, test_size_reg)
    
    if X_train is None:
        st.error("Erro ao carregar dados")
    else:
        # Modelo Linear
        with model_tabs[0]:
            st.subheader("📊 Regressão Linear")
            with st.spinner("Treinando modelo Linear..."):
                model = RegressionModel(model_type='linear')
                model.train(X_train, y_train)
                y_pred = model.predict(X_test)
                y_test_float = y_test.astype(float)
                y_pred_float = np.array(y_pred, dtype=float)
                residuals = y_test_float - y_pred_float
            
            metrics = model.evaluate(X_test, y_test)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R²", f"{metrics['R²']:.4f}")
            col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
            col3.metric("MAE", f"{metrics['MAE']:.2f}")
            col4.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = plot_actual_vs_predicted(y_test_float.values, y_pred_float)
                st.plotly_chart(fig, use_container_width=True, key='linear_actual_pred')
            with col2:
                fig = plot_residuals(residuals.values if hasattr(residuals, 'values') else residuals, y_pred_float)
                st.plotly_chart(fig, use_container_width=True, key='linear_residuals')
        
        # Modelo Random Forest
        with model_tabs[1]:
            st.subheader("📊 Random Forest")
            with st.spinner("Treinando modelo Random Forest..."):
                model = RegressionModel(model_type='random_forest')
                model.train(X_train, y_train)
                y_pred = model.predict(X_test)
                y_test_float = y_test.astype(float)
                y_pred_float = np.array(y_pred, dtype=float)
                residuals = y_test_float - y_pred_float
            
            metrics = model.evaluate(X_test, y_test)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R²", f"{metrics['R²']:.4f}")
            col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
            col3.metric("MAE", f"{metrics['MAE']:.2f}")
            col4.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = plot_actual_vs_predicted(y_test_float.values, y_pred_float)
                st.plotly_chart(fig, use_container_width=True, key='rf_actual_pred')
            with col2:
                importance_df = model.get_feature_importance()
                if importance_df is not None:
                    fig = plot_feature_importance(importance_df)
                    st.plotly_chart(fig, use_container_width=True, key='rf_importance')
        
        # Modelo XGBoost
        with model_tabs[2]:
            st.subheader("📊 XGBoost")
            with st.spinner("Treinando modelo XGBoost..."):
                model = RegressionModel(model_type='xgboost')
                model.train(X_train, y_train)
                y_pred = model.predict(X_test)
                y_test_float = y_test.astype(float)
                y_pred_float = np.array(y_pred, dtype=float)
                residuals = y_test_float - y_pred_float
            
            metrics = model.evaluate(X_test, y_test)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R²", f"{metrics['R²']:.4f}")
            col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
            col3.metric("MAE", f"{metrics['MAE']:.2f}")
            col4.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = plot_actual_vs_predicted(y_test_float.values, y_pred_float)
                st.plotly_chart(fig, use_container_width=True, key='xgb_actual_pred')
            with col2:
                importance_df = model.get_feature_importance()
                if importance_df is not None:
                    fig = plot_feature_importance(importance_df)
                    st.plotly_chart(fig, use_container_width=True, key='xgb_importance')
        
        # Comparação de Modelos
        with model_tabs[3]:
            st.subheader("📊 Comparação dos 3 Modelos")
            
            with st.spinner("Treinando e avaliando modelos..."):
                results_df = compare_models(
                    X_train, X_test, y_train, y_test,
                    models_to_compare=['linear', 'random_forest', 'xgboost']
                )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(
                    results_df.style.format({
                        'R²': '{:.4f}',
                        'RMSE': '{:.2f}',
                        'MAE': '{:.2f}',
                        'MAPE': '{:.2f}%'
                    }).background_gradient(cmap='RdYlGn', subset=['R²']),
                    use_container_width=True
                )
            
            with col2:
                fig = plot_metrics_comparison(
                    results_df,
                    metric_cols=['R²', 'RMSE', 'MAE'],
                    title="Comparação de Métricas"
                )
                st.plotly_chart(fig, use_container_width=True, key='reg_comparison')

with tab2:
    st.header("🎯 Modelos de Classificação")
    st.markdown("Classificação do desempenho em **categorias** (Baixo/Médio/Alto)")
    
    st.info("""📊 **Análise Ecológica (Nível Municipal)**  
    - Classificação de **municípios** em categorias de desempenho
    - Classes baseadas em quantis da média municipal
    - ⚠️ Análise agregada, não prevê desempenho individual
    """)
    
    # Configurações na sidebar
    st.sidebar.subheader("⚙️ Configurações - Classificação")
    
    sample_size_clf = st.sidebar.slider(
        "Número de municípios",
        min_value=100,
        max_value=5500,
        value=3000,
        step=500,
        key='sample_clf',
        help="Total de municípios disponíveis: ~5.570"
    )
    
    n_classes = st.sidebar.selectbox(
        "Número de classes",
        options=[2, 3],
        index=1,
        key='n_classes'
    )
    
    test_size_clf = st.sidebar.slider(
        "% Teste",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
        key='test_size_clf'
    )
    
    # Criar abas para modelos individuais
    clf_tabs = st.tabs([
        "📊 Logistic",
        "📊 Random Forest",
        "📊 XGBoost",
        "📉 Comparação"
    ])
    
    # Preparar dados uma única vez
    @st.cache_data(ttl=3600)
    def get_classification_data(sample_size, n_cls, test_size):
        """Carrega dados MUNICIPAIS para classificação"""
        df_municipios = load_municipio_data(min_participantes=30)
        
        if df_municipios.empty:
            return None, None, None, None, None
        
        # Features municipais
        feature_cols = [
            'perc_pai_ensino_superior',
            'perc_mae_ensino_superior',
            'perc_renda_alta',
            'perc_renda_baixa',
            'media_pessoas_residencia',
            'perc_escola_publica'
        ]
        
        # Amostrar municípios
        if len(df_municipios) > sample_size:
            df_sample = df_municipios.sample(n=min(sample_size, len(df_municipios)), random_state=42)
        else:
            df_sample = df_municipios
        
        return prepare_classification_data(
            df_sample,
            target_col='media_geral',
            feature_cols=feature_cols,
            n_classes=n_cls,
            test_size=test_size
        )
    
    X_train, X_test, y_train, y_test, class_labels = get_classification_data(sample_size_clf, n_classes, test_size_clf)
    
    if X_train is None:
        st.error("Erro ao carregar dados")
    else:
        st.info(f"**Classes:** {', '.join(class_labels)}")
        
        # Logistic Regression
        with clf_tabs[0]:
            st.subheader("📊 Logistic Regression")
            with st.spinner("Treinando modelo Logistic..."):
                model = ClassificationModel(model_type='logistic')
                model.train(X_train, y_train)
            
            metrics = model.evaluate(X_test, y_test)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['precision']:.4f}")
            col3.metric("Recall", f"{metrics['recall']:.4f}")
            col4.metric("F1-Score", f"{metrics['f1']:.4f}")
            
            col1, col2 = st.columns(2)
            with col1:
                cm = model.get_confusion_matrix(X_test, y_test)
                fig = plot_confusion_matrix(cm, class_labels)
                st.plotly_chart(fig, use_container_width=True, key='log_confusion')
            with col2:
                if n_classes == 2:
                    roc_data = model.get_roc_curve_data(X_test, y_test)
                    if roc_data:
                        fig = plot_roc_curve(roc_data)
                        st.plotly_chart(fig, use_container_width=True, key='log_roc')
        
        # Random Forest
        with clf_tabs[1]:
            st.subheader("📊 Random Forest Classifier")
            with st.spinner("Treinando modelo Random Forest..."):
                model = ClassificationModel(model_type='random_forest')
                model.train(X_train, y_train)
            
            metrics = model.evaluate(X_test, y_test)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['precision']:.4f}")
            col3.metric("Recall", f"{metrics['recall']:.4f}")
            col4.metric("F1-Score", f"{metrics['f1']:.4f}")
            
            col1, col2 = st.columns(2)
            with col1:
                cm = model.get_confusion_matrix(X_test, y_test)
                fig = plot_confusion_matrix(cm, class_labels)
                st.plotly_chart(fig, use_container_width=True, key='rf_clf_confusion')
            with col2:
                importance_df = model.get_feature_importance()
                if importance_df is not None:
                    fig = plot_feature_importance(importance_df)
                    st.plotly_chart(fig, use_container_width=True, key='rf_clf_importance')
        
        # XGBoost
        with clf_tabs[2]:
            st.subheader("📊 XGBoost Classifier")
            with st.spinner("Treinando modelo XGBoost..."):
                model = ClassificationModel(model_type='xgboost')
                model.train(X_train, y_train)
            
            metrics = model.evaluate(X_test, y_test)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['precision']:.4f}")
            col3.metric("Recall", f"{metrics['recall']:.4f}")
            col4.metric("F1-Score", f"{metrics['f1']:.4f}")
            
            col1, col2 = st.columns(2)
            with col1:
                cm = model.get_confusion_matrix(X_test, y_test)
                fig = plot_confusion_matrix(cm, class_labels)
                st.plotly_chart(fig, use_container_width=True, key='xgb_clf_confusion')
            with col2:
                importance_df = model.get_feature_importance()
                if importance_df is not None:
                    fig = plot_feature_importance(importance_df)
                    st.plotly_chart(fig, use_container_width=True, key='xgb_clf_importance')
        
        # Comparação
        with clf_tabs[3]:
            st.subheader("📊 Comparação dos 3 Modelos")
            
            with st.spinner("Treinando e avaliando modelos..."):
                results_df = compare_classifiers(
                    X_train, X_test, y_train, y_test,
                    models_to_compare=['logistic', 'random_forest', 'xgboost']
                )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(
                    results_df.style.format({
                        'Accuracy': '{:.4f}',
                        'Precision': '{:.4f}',
                        'Recall': '{:.4f}',
                        'F1-Score': '{:.4f}'
                    }).background_gradient(cmap='RdYlGn', subset=['F1-Score']),
                    use_container_width=True
                )
            
            with col2:
                fig = plot_metrics_comparison(
                    results_df,
                    metric_cols=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    title="Comparação de Métricas"
                )
                st.plotly_chart(fig, use_container_width=True, key='clf_comparison')

with tab3:
    st.header("👥 Modelos de Clustering")
    st.markdown("Agrupamento de **perfis socioeconômicos** similares")
    
    st.info("""📊 **Análise de Perfis Municipais**  
    - Agrupa **municípios** com características socioeconômicas similares
    - Identifica padrões regionais e clusters de desempenho
    - Útil para políticas públicas direcionadas por perfil municipal
    """)
    
    # Configurações na sidebar
    st.sidebar.subheader("⚙️ Configurações - Clustering")
    
    sample_size_clust = st.sidebar.slider(
        "Número de municípios",
        min_value=100,
        max_value=5500,
        value=2000,
        step=500,
        key='sample_clust',
        help="Total de municípios disponíveis: ~5.570"
    )
    
    n_clusters = st.sidebar.slider(
        "Número de clusters",
        min_value=2,
        max_value=10,
        value=3,
        key='n_clusters'
    )
    
    cluster_method = st.sidebar.selectbox(
        "Método de clustering",
        options=['kmeans', 'hierarchical'],
        index=0,
        key='cluster_method'
    )
    
    find_k = st.sidebar.checkbox("Encontrar k ótimo (Método do Cotovelo)", value=False)
    
    # Carregar e processar dados
    @st.cache_data(ttl=3600)
    def get_clustering_data(sample_size):
        """Carrega dados MUNICIPAIS para clustering"""
        df_municipios = load_municipio_data(min_participantes=30)
        
        if df_municipios.empty:
            return pd.DataFrame()
        
        # Amostrar municípios
        if len(df_municipios) > sample_size:
            return df_municipios.sample(n=min(sample_size, len(df_municipios)), random_state=42)
        else:
            return df_municipios
    
    with st.spinner("Carregando dados..."):
        df_clust = get_clustering_data(sample_size_clust)
    
    if df_clust.empty:
        st.error("Erro ao carregar dados")
    else:
        st.success(f"✅ Dados carregados: {len(df_clust):,} municípios")
        
        # Features municipais (percentuais/médias)
        feature_cols = [
            'perc_pai_ensino_superior',
            'perc_mae_ensino_superior',
            'perc_renda_alta',
            'perc_renda_baixa',
            'media_pessoas_residencia',
            'perc_escola_publica'
        ]
        
        # Preparar dados: remover NaNs e converter para float
        X = df_clust[feature_cols].dropna()
        X_encoded = X.astype(float)  # Dados municipais já são numéricos
        
        y = df_clust.loc[X_encoded.index, 'media_geral'].astype(float)
        
        st.info(f"📊 Dados preparados: {len(X_encoded):,} municípios válidos")
        
        if find_k:
            st.markdown("---")
            st.subheader("🔍 Encontrando k Ótimo")
            
            with st.spinner("Testando diferentes valores de k..."):
                elbow_df = find_optimal_k(X_encoded, k_range=range(2, 11))
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = plot_elbow_curve(elbow_df)
                st.plotly_chart(fig, use_container_width=True, key='clust_elbow')
            
            with col2:
                fig = plot_silhouette_scores(elbow_df)
                st.plotly_chart(fig, use_container_width=True, key='clust_silhouette')
            
            best_k_silhouette = elbow_df.loc[elbow_df['silhouette'].idxmax(), 'k']
            st.info(f"💡 **Sugestão:** k = {best_k_silhouette} (maior score de silhueta)")
            
            n_clusters = int(best_k_silhouette)
        
        st.markdown("---")
        st.subheader(f"📊 Clustering com k = {n_clusters}")
        
        with st.spinner("Executando clustering..."):
            model = ClusteringModel(
                model_type=cluster_method,
                n_clusters=n_clusters
            )
            model.fit(X_encoded)
            metrics = model.evaluate(X_encoded)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Clusters", metrics['n_clusters'])
        
        with col2:
            if metrics['silhouette_score']:
                st.metric("Score de Silhueta", f"{metrics['silhouette_score']:.3f}")
        
        with col3:
            if metrics['davies_bouldin_score']:
                st.metric("Davies-Bouldin", f"{metrics['davies_bouldin_score']:.3f}")
        
        st.markdown("---")
        st.subheader("🎨 Visualização dos Clusters")
        
        with st.spinner("Reduzindo dimensionalidade (PCA)..."):
            df_pca_2d = model.reduce_dimensions(X_encoded, n_components=2)
        
        fig = plot_cluster_scatter_2d(df_pca_2d)
        st.plotly_chart(fig, use_container_width=True, key='clust_scatter')
        
        st.markdown("---")
        st.subheader("📋 Perfil dos Clusters")
        
        profile_df = model.get_cluster_profile(X_encoded, y)
        
        st.dataframe(
            profile_df[['Cluster', 'N', 'Proporção', 'target_mean', 'target_std']].style.format({
                'Proporção': '{:.1f}%',
                'target_mean': '{:.1f}',
                'target_std': '{:.1f}'
            }).background_gradient(cmap='Blues', subset=['target_mean']),
            use_container_width=True
        )
        
        st.markdown("##### Características Médias por Cluster")
        
        value_cols = [col for col in profile_df.columns if col.endswith('_mean') and col != 'target_mean']
        
        if value_cols:
            fig = plot_cluster_profile_heatmap(
                profile_df,
                value_cols=value_cols[:6]
            )
            st.plotly_chart(fig, use_container_width=True, key='clust_heatmap')
        
        if cluster_method == 'kmeans':
            st.markdown("##### Centros dos Clusters")
            centers_df = model.get_cluster_centers()
            if centers_df is not None:
                st.dataframe(
                    centers_df.style.format('{:.2f}').background_gradient(cmap='YlOrRd', axis=1),
                    use_container_width=True
                )
        
        st.success("✅ Análise de clustering concluída!")

st.markdown("---")
st.info("""
**💡 Interpretação dos Modelos:**
- **Regressão:** R² próximo de 1 indica boa capacidade preditiva
- **Classificação:** F1-Score equilibra precisão e recall
- **Clustering:** Silhuette Score > 0.5 indica boa separação dos clusters
""")
