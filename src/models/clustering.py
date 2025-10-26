"""
Modelos de Clustering para agrupar perfis de estudantes
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import streamlit as st


class ClusteringModel:
    """Classe base para modelos de clustering"""
    
    def __init__(self, model_type: str = 'kmeans', n_clusters: int = 3,
                 random_state: int = 42):
        """
        Args:
            model_type: Tipo do modelo ('kmeans', 'dbscan', 'hierarchical')
            n_clusters: Número de clusters (para kmeans e hierarchical)
            random_state: Seed para reprodutibilidade
        """
        self.model_type = model_type
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_names = None
        self.labels_ = None
        self.metrics = {}
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa o modelo baseado no tipo"""
        models = {
            'kmeans': KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            ),
            'dbscan': DBSCAN(
                eps=0.5,
                min_samples=5
            ),
            'hierarchical': AgglomerativeClustering(
                n_clusters=self.n_clusters
            )
        }
        
        self.model = models.get(self.model_type, KMeans(n_clusters=self.n_clusters))
    
    @st.cache_data(ttl=3600)
    def fit(_self, X: pd.DataFrame, scale_features: bool = True):
        """
        Treina o modelo de clustering
        
        Args:
            X: Features
            scale_features: Se True, normaliza as features
        """
        _self.feature_names = X.columns.tolist()
        
        if scale_features:
            X_scaled = _self.scaler.fit_transform(X)
        else:
            X_scaled = X.values
        
        _self.labels_ = _self.model.fit_predict(X_scaled)
        
        return _self
    
    def predict(self, X: pd.DataFrame, scale_features: bool = True):
        """
        Prediz clusters para novos dados (apenas para KMeans)
        """
        if self.model_type != 'kmeans':
            raise ValueError("Predict só está disponível para KMeans")
        
        if scale_features:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        return self.model.predict(X_scaled)
    
    def evaluate(self, X: pd.DataFrame, scale_features: bool = True):
        """
        Avalia a qualidade do clustering
        
        Returns:
            dict com métricas
        """
        if scale_features:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        n_unique_labels = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        
        if n_unique_labels < 2:
            return {
                'n_clusters': n_unique_labels,
                'silhouette_score': None,
                'calinski_harabasz_score': None,
                'davies_bouldin_score': None,
                'noise_points': np.sum(self.labels_ == -1)
            }
        
        self.metrics = {
            'n_clusters': n_unique_labels,
            'silhouette_score': silhouette_score(X_scaled, self.labels_),
            'calinski_harabasz_score': calinski_harabasz_score(X_scaled, self.labels_),
            'davies_bouldin_score': davies_bouldin_score(X_scaled, self.labels_),
            'noise_points': np.sum(self.labels_ == -1)
        }
        
        return self.metrics
    
    def get_cluster_centers(self):
        """
        Retorna centros dos clusters (apenas KMeans)
        
        Returns:
            DataFrame com centros dos clusters
        """
        if self.model_type != 'kmeans':
            return None
        
        centers = self.scaler.inverse_transform(self.model.cluster_centers_)
        
        return pd.DataFrame(
            centers,
            columns=self.feature_names,
            index=[f'Cluster {i}' for i in range(self.n_clusters)]
        )
    
    def get_cluster_profile(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Cria perfil de cada cluster
        
        Args:
            X: Features originais
            y: Target opcional (ex: nota_media)
        
        Returns:
            DataFrame com estatísticas por cluster
        """
        df_profile = X.copy()
        df_profile['cluster'] = self.labels_
        
        if y is not None:
            df_profile['target'] = y.values
        
        profiles = []
        
        for cluster in sorted(df_profile['cluster'].unique()):
            if cluster == -1:
                cluster_name = 'Ruído'
            else:
                cluster_name = f'Cluster {cluster}'
            
            cluster_data = df_profile[df_profile['cluster'] == cluster]
            
            profile = {
                'Cluster': cluster_name,
                'N': len(cluster_data),
                'Proporção': len(cluster_data) / len(df_profile) * 100
            }
            
            for col in self.feature_names:
                profile[f'{col}_mean'] = cluster_data[col].mean()
                profile[f'{col}_std'] = cluster_data[col].std()
            
            if y is not None:
                profile['target_mean'] = cluster_data['target'].mean()
                profile['target_std'] = cluster_data['target'].std()
            
            profiles.append(profile)
        
        return pd.DataFrame(profiles)
    
    def reduce_dimensions(self, X: pd.DataFrame, n_components: int = 2,
                           scale_features: bool = True):
        """
        Reduz dimensionalidade para visualização
        
        Args:
            X: Features
            n_components: Número de componentes PCA (2 ou 3)
            scale_features: Se True, normaliza antes do PCA
        
        Returns:
            DataFrame com componentes principais e labels
        """
        if scale_features:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X_scaled)
        
        df_pca = pd.DataFrame(
            X_pca,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        df_pca['cluster'] = self.labels_
        
        return df_pca
    
    def get_pca_explained_variance(self):
        """Retorna variância explicada por cada componente PCA"""
        if self.pca is None:
            return None
        
        return pd.DataFrame({
            'Componente': [f'PC{i+1}' for i in range(len(self.pca.explained_variance_ratio_))],
            'Variância Explicada': self.pca.explained_variance_ratio_,
            'Variância Acumulada': np.cumsum(self.pca.explained_variance_ratio_)
        })


@st.cache_data(ttl=3600)
def find_optimal_k(X: pd.DataFrame, k_range: range = range(2, 11),
                   scale_features: bool = True, random_state: int = 42):
    """
    Método do cotovelo para encontrar k ótimo
    
    Args:
        X: Features
        k_range: Range de valores de k para testar
        scale_features: Se True, normaliza as features
        random_state: Seed
    
    Returns:
        DataFrame com métricas para cada k
    """
    scaler = StandardScaler()
    
    if scale_features:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    
    results = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        results.append({
            'k': k,
            'inertia': kmeans.inertia_,
            'silhouette': silhouette_score(X_scaled, labels),
            'calinski_harabasz': calinski_harabasz_score(X_scaled, labels),
            'davies_bouldin': davies_bouldin_score(X_scaled, labels)
        })
    
    return pd.DataFrame(results)


def compare_clustering_methods(X: pd.DataFrame, n_clusters: int = 3,
                                 scale_features: bool = True, random_state: int = 42):
    """
    Compara diferentes métodos de clustering
    
    Args:
        X: Features
        n_clusters: Número de clusters
        scale_features: Se True, normaliza as features
        random_state: Seed
    
    Returns:
        DataFrame com métricas de cada método
    """
    methods = ['kmeans', 'hierarchical']
    results = []
    
    for method in methods:
        model = ClusteringModel(
            model_type=method,
            n_clusters=n_clusters,
            random_state=random_state
        )
        model.fit(X, scale_features)
        metrics = model.evaluate(X, scale_features)
        
        results.append({
            'Método': method,
            **metrics
        })
    
    return pd.DataFrame(results)
