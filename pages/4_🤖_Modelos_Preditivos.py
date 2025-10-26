"""
Página 4 - Modelos Preditivos
Interface para treinamento e avaliação de modelos de Machine Learning
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.utils.config import Config
from src.utils.constants import *
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

st.title("🤖 Modelos Preditivos")
st.markdown("""
Aplicação de Machine Learning para análise e predição de desempenho no ENEM 2024.
""")
st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "📈 Regressão",
    "🎯 Classificação",
    "👥 Clustering"
])

with tab1:
    st.header("📈 Modelos de Regressão")
    st.markdown("Predição da **nota média** baseada em fatores socioeconômicos")
    
    with st.sidebar:
        st.subheader("⚙️ Configurações - Regressão")
        
        sample_size_reg = st.slider(
            "Tamanho da amostra",
            min_value=10000,
            max_value=200000,
            value=100000,
            step=10000,
            key='sample_reg'
        )
        
        test_size_reg = st.slider(
            "% Teste",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            key='test_size_reg'
        )
        
        models_to_compare_reg = st.multiselect(
            "Modelos para comparar",
            options=['linear', 'ridge', 'lasso', 'random_forest', 'gradient_boosting', 'xgboost'],
            default=['linear', 'random_forest', 'xgboost'],
            key='models_reg'
        )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Features Selecionadas")
        st.markdown("""
        **Variáveis Independentes (Features):**
        - Escolaridade do Pai
        - Escolaridade da Mãe
        - Ocupação do Pai
        - Ocupação da Mãe
        - Faixa de Renda
        - Pessoas na Residência
        
        **Variável Dependente (Target):**
        - Nota Média (5 provas)
        """)
    
    with col2:
        st.subheader("📊 Tamanho dos Dados")
        total_train = int(sample_size_reg * (1 - test_size_reg))
        total_test = int(sample_size_reg * test_size_reg)
        
        st.metric("Total", f"{sample_size_reg:,}")
        st.metric("Treino", f"{total_train:,}")
        st.metric("Teste", f"{total_test:,}")
    
    if st.button("🚀 Treinar Modelos de Regressão", key='train_reg'):
        with st.spinner("Carregando dados..."):
            df_reg = get_participantes_sample(
                limit=sample_size_reg,
                columns=[
                    'escolaridade_pai', 'escolaridade_mae',
                    'ocupacao_pai', 'ocupacao_mae',
                    'faixa_renda', 'pessoas_residencia',
                    'nota_media_5_notas'
                ],
                where_clause="nota_media_5_notas IS NOT NULL"
            )
            
            if not df_reg.empty:
                st.success(f"✅ Dados carregados: {len(df_reg):,} registros")
                
                feature_cols = [
                    'escolaridade_pai', 'escolaridade_mae',
                    'ocupacao_pai', 'ocupacao_mae',
                    'faixa_renda', 'pessoas_residencia'
                ]
                
                with st.spinner("Preparando dados..."):
                    X_train, X_test, y_train, y_test = prepare_regression_data(
                        df_reg,
                        target_col='nota_media_5_notas',
                        feature_cols=feature_cols,
                        test_size=test_size_reg
                    )
                
                st.markdown("---")
                st.subheader("📊 Comparação de Modelos")
                
                with st.spinner("Treinando e avaliando modelos..."):
                    results_df = compare_models(
                        X_train, X_test, y_train, y_test,
                        models_to_compare=models_to_compare_reg
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
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.subheader("🔍 Análise Detalhada do Melhor Modelo")
                
                best_model_name = results_df.iloc[0]['Modelo']
                st.info(f"**Melhor Modelo:** {best_model_name} (R² = {results_df.iloc[0]['R²']:.4f})")
                
                with st.spinner(f"Treinando {best_model_name}..."):
                    best_model = RegressionModel(model_type=best_model_name)
                    best_model.train(X_train, y_train)
                    y_pred = best_model.predict(X_test)
                    residuals = y_test - y_pred
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = plot_actual_vs_predicted(y_test.values, y_pred)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = plot_residuals(residuals.values, y_pred)
                    st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = plot_residuals_distribution(residuals.values)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    importance_df = best_model.get_feature_importance()
                    if importance_df is not None:
                        fig = plot_feature_importance(importance_df)
                        st.plotly_chart(fig, use_container_width=True)
                
                st.success("✅ Análise de regressão concluída!")

with tab2:
    st.header("🎯 Modelos de Classificação")
    st.markdown("Classificação do desempenho em categorias: **Baixo**, **Médio**, **Alto**")
    
    with st.sidebar:
        st.subheader("⚙️ Configurações - Classificação")
        
        sample_size_clf = st.slider(
            "Tamanho da amostra",
            min_value=10000,
            max_value=200000,
            value=100000,
            step=10000,
            key='sample_clf'
        )
        
        n_classes = st.selectbox(
            "Número de classes",
            options=[2, 3],
            index=1,
            key='n_classes'
        )
        
        test_size_clf = st.slider(
            "% Teste",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            key='test_size_clf'
        )
        
        models_to_compare_clf = st.multiselect(
            "Modelos para comparar",
            options=['logistic', 'random_forest', 'gradient_boosting', 'xgboost'],
            default=['logistic', 'random_forest', 'xgboost'],
            key='models_clf'
        )
    
    if st.button("🚀 Treinar Modelos de Classificação", key='train_clf'):
        with st.spinner("Carregando dados..."):
            df_clf = get_participantes_sample(
                limit=sample_size_clf,
                columns=[
                    'escolaridade_pai', 'escolaridade_mae',
                    'ocupacao_pai', 'ocupacao_mae',
                    'faixa_renda', 'pessoas_residencia',
                    'nota_media_5_notas'
                ],
                where_clause="nota_media_5_notas IS NOT NULL"
            )
            
            if not df_clf.empty:
                st.success(f"✅ Dados carregados: {len(df_clf):,} registros")
                
                feature_cols = [
                    'escolaridade_pai', 'escolaridade_mae',
                    'ocupacao_pai', 'ocupacao_mae',
                    'faixa_renda', 'pessoas_residencia'
                ]
                
                with st.spinner("Preparando dados e criando classes..."):
                    X_train, X_test, y_train, y_test, class_labels = prepare_classification_data(
                        df_clf,
                        target_col='nota_media_5_notas',
                        feature_cols=feature_cols,
                        n_classes=n_classes,
                        test_size=test_size_clf
                    )
                
                st.info(f"**Classes criadas:** {', '.join(class_labels)}")
                
                st.markdown("---")
                st.subheader("📊 Comparação de Modelos")
                
                with st.spinner("Treinando e avaliando modelos..."):
                    results_df = compare_classifiers(
                        X_train, X_test, y_train, y_test,
                        models_to_compare=models_to_compare_clf
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
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.subheader("🔍 Análise Detalhada do Melhor Modelo")
                
                best_model_name = results_df.iloc[0]['Modelo']
                st.info(f"**Melhor Modelo:** {best_model_name} (F1-Score = {results_df.iloc[0]['F1-Score']:.4f})")
                
                with st.spinner(f"Treinando {best_model_name}..."):
                    best_model = ClassificationModel(model_type=best_model_name)
                    best_model.train(X_train, y_train)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Matriz de Confusão")
                    cm = best_model.get_confusion_matrix(X_test, y_test)
                    fig = plot_confusion_matrix(cm, class_labels)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("##### Importância das Features")
                    importance_df = best_model.get_feature_importance()
                    if importance_df is not None:
                        fig = plot_feature_importance(importance_df)
                        st.plotly_chart(fig, use_container_width=True)
                
                if n_classes == 2:
                    st.markdown("##### Curva ROC")
                    roc_data = best_model.get_roc_curve_data(X_test, y_test)
                    if roc_data:
                        fig = plot_roc_curve(roc_data)
                        st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("##### Relatório de Classificação")
                report = best_model.get_classification_report(X_test, y_test)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(
                    report_df.style.format('{:.3f}').background_gradient(cmap='Blues'),
                    use_container_width=True
                )
                
                st.success("✅ Análise de classificação concluída!")

with tab3:
    st.header("👥 Modelos de Clustering")
    st.markdown("Agrupamento de estudantes com perfis socioeconômicos similares")
    
    with st.sidebar:
        st.subheader("⚙️ Configurações - Clustering")
        
        sample_size_clust = st.slider(
            "Tamanho da amostra",
            min_value=10000,
            max_value=100000,
            value=50000,
            step=10000,
            key='sample_clust'
        )
        
        find_k = st.checkbox("Encontrar k ótimo (Método do Cotovelo)", value=True)
        
        if not find_k:
            n_clusters = st.slider(
                "Número de clusters",
                min_value=2,
                max_value=10,
                value=3,
                key='n_clusters'
            )
        
        cluster_method = st.selectbox(
            "Método de clustering",
            options=['kmeans', 'hierarchical'],
            index=0,
            key='cluster_method'
        )
    
    if st.button("🚀 Executar Clustering", key='train_clust'):
        with st.spinner("Carregando dados..."):
            df_clust = get_participantes_sample(
                limit=sample_size_clust,
                columns=[
                    'escolaridade_pai', 'escolaridade_mae',
                    'ocupacao_pai', 'ocupacao_mae',
                    'faixa_renda', 'pessoas_residencia',
                    'nota_media_5_notas'
                ],
                where_clause="nota_media_5_notas IS NOT NULL"
            )
            
            if not df_clust.empty:
                st.success(f"✅ Dados carregados: {len(df_clust):,} registros")
                
                feature_cols = [
                    'escolaridade_pai', 'escolaridade_mae',
                    'ocupacao_pai', 'ocupacao_mae',
                    'faixa_renda', 'pessoas_residencia'
                ]
                
                X = df_clust[feature_cols].dropna()
                y = df_clust.loc[X.index, 'nota_media_5_notas']
                
                if find_k:
                    st.markdown("---")
                    st.subheader("🔍 Encontrando k Ótimo")
                    
                    with st.spinner("Testando diferentes valores de k..."):
                        elbow_df = find_optimal_k(X, k_range=range(2, 11))
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = plot_elbow_curve(elbow_df)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = plot_silhouette_scores(elbow_df)
                        st.plotly_chart(fig, use_container_width=True)
                    
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
                    model.fit(X)
                    metrics = model.evaluate(X)
                
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
                    df_pca_2d = model.reduce_dimensions(X, n_components=2)
                
                fig = plot_cluster_scatter_2d(df_pca_2d)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.subheader("📋 Perfil dos Clusters")
                
                profile_df = model.get_cluster_profile(X, y)
                
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
                    st.plotly_chart(fig, use_container_width=True)
                
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
