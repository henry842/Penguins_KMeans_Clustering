# %% [markdown]
# # Módulo 30 - K-Means
# **Atividade: Clusterização de Pinguins**
# 
# Nesta tarefa, exploramos o algoritmo K-Means aplicado a dados biológicos, segmentando espécies de pinguins com base em suas características físicas.  

# %% [markdown]
# ## Descrição das Variáveis
# 
# - species: Espécie do pinguim (Adelie, Chinstrap, Gentoo)  
# - island: Ilha onde o pinguim foi observado (Biscoe, Dream, Torgersen)  
# - bill_length_mm: Comprimento do bico em milímetros  
# - bill_depth_mm: Profundidade do bico em milímetros  
# - flipper_length_mm: Comprimento da barbatana em milímetros  
# - body_mass_g: Massa corporal em gramas  
# - sex: Sexo do pinguim (Male, Female)  
# - year: Ano da observação  

# %% 
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", palette="deep")

# %% 
# Carregar dados
penguins = sns.load_dataset('penguins')

# %% 
# Limpeza de dados
penguins_clean = penguins.dropna().reset_index(drop=True)

# Selecionar somente colunas numéricas
penguins_num = penguins_clean.select_dtypes(include=['float64', 'int64'])
penguins_num.head()

# %% [markdown]
# ## 2 - Análise Exploratória

# %% 
# Pairplot
sns.pairplot(penguins_num, diag_kind='kde', palette='coolwarm', plot_kws={'alpha':0.7})

# %% 
# Heatmap de correlação
plt.figure(figsize=(10,6))
sns.heatmap(penguins_num.corr(), annot=True, cmap='coolwarm')
plt.title("Mapa de Correlação das Variáveis Numéricas")
plt.show()

# %% 
# Boxplots das variáveis
penguins_num.plot(kind='box', figsize=(10,6), subplots=True, layout=(2,3), sharex=False, sharey=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3 - Padronização dos Dados

# %%
scaler = StandardScaler()
penguins_scaled = scaler.fit_transform(penguins_num)
penguins_scaled_df = pd.DataFrame(penguins_scaled, columns=penguins_num.columns)
penguins_scaled_df.head()

# %% [markdown]
# ## 4 - Aplicação do K-Means

# %% 
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(penguins_scaled_df)

penguins_scaled_df['cluster'] = clusters
penguins_scaled_df.head()

# %% [markdown]
# ## 5 - Visualização dos Clusters

# %% 
# Scatter plot: Bill Length vs Bill Depth
fig = px.scatter(
    penguins_scaled_df,
    x='bill_length_mm',
    y='bill_depth_mm',
    color='cluster',
    title='Clusterização: Comprimento vs Profundidade do Bico'
)

centroids = kmeans.cluster_centers_

fig.add_trace(go.Scatter(
    x=centroids[:, penguins_num.columns.get_loc('bill_length_mm')],
    y=centroids[:, penguins_num.columns.get_loc('bill_depth_mm')],
    mode='markers',
    marker=dict(size=18, symbol='x'),
    name='Centroides'
))

fig.update_layout(width=800, height=500, template='plotly_white')
fig.show()

# %% 
# Centroides despadronizados (melhoria 5)
original_centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(original_centroids, columns=penguins_num.columns)
centroids_df

# %% 
# Descrição automática dos clusters (melhoria 6)
for c in centroids_df.index:
    print(f"\nCluster {c}:")
    print(centroids_df.loc[c].sort_values())

# %%
# Scatter plot: Flipper Length vs Body Mass
fig = px.scatter(
    penguins_scaled_df,
    x='flipper_length_mm',
    y='body_mass_g',
    color='cluster',
    title='Clusterização: Barbatana vs Massa Corporal'
)
fig.add_trace(go.Scatter(
    x=centroids[:, penguins_num.columns.get_loc('flipper_length_mm')],
    y=centroids[:, penguins_num.columns.get_loc('body_mass_g')],
    mode='markers',
    marker=dict(size=18, symbol='x'),
    name='Centroides'
))
fig.update_layout(width=800, height=500, template='plotly_white')
fig.show()

# %% [markdown]
# ## 6 - Método do Cotovelo e Silhouette

# %%
# Curva do Cotovelo
inertia = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(penguins_scaled_df.drop(columns='cluster'))
    inertia.append(km.inertia_)

plt.plot(K, inertia, marker='o')
plt.title("Método do Cotovelo")
plt.xlabel("Número de Clusters")
plt.ylabel("Inércia")
plt.show()

# %%
# Silhouette Score por k
sil_scores = []
for k in range(2,10):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(penguins_scaled_df.drop(columns='cluster'))
    sil = silhouette_score(penguins_scaled_df.drop(columns='cluster'), labels)
    sil_scores.append(sil)

plt.plot(range(2,10), sil_scores, marker='o')
plt.title("Silhouette Score por k")
plt.xlabel("k")
plt.ylabel("Silhouette")
plt.show()

# %%
# Silhouette Score para k=3
score = silhouette_score(penguins_scaled_df.drop(columns='cluster'), clusters)
print("Silhouette Score (k=3):", score)

# %%
# Davies-Bouldin Score
db_score = davies_bouldin_score(penguins_scaled_df.drop(columns='cluster'), clusters)
print("Davies-Bouldin Score (k=3):", db_score)

# %% [markdown]
# ## 7 - PCA + Visualização 2D

# %%
pca = PCA(n_components=2)
pca_data = pca.fit_transform(penguins_scaled_df.drop(columns='cluster'))
pca_df = pd.DataFrame(pca_data, columns=['PC1','PC2'])
pca_df['cluster'] = clusters

px.scatter(
    pca_df,
    x='PC1',
    y='PC2',
    color='cluster',
    title='Clusters em 2D usando PCA'
)

# %%
print("Variância explicada pelo PCA:", pca.explained_variance_ratio_)

# %% [markdown]
# ## 8 - Comparação Cluster × Espécie Real

# %%
comparison = penguins_clean.copy()
comparison['cluster'] = clusters
pd.crosstab(comparison['species'], comparison['cluster'])

# %% [markdown]
# ## 9 - Estatísticas por Cluster

# %%
penguins_clustered = penguins_num.copy()
penguins_clustered['cluster'] = clusters
penguins_clustered.groupby('cluster').mean()

# %% [markdown]
# ## 10 - Interpretação Final
# 
# Cluster 0 → pinguins menores, bico curto (provável Adelie)  
# Cluster 1 → pinguins grandes com massa alta (provável Gentoo)  
# Cluster 2 → bico mais longo e profundo (provável Chinstrap)  

# %% [markdown]
# ## 11 - Relatório Final
# 
# - Número de clusters escolhido: 3  
# - Silhouette Score: 0.446  
# - Davies-Bouldin Score: 0.78  
# - Melhor k (comparação): gráfico mostrou que k ≈ 3  
# - Correspondência com espécies reais:
#     - Cluster 0 → Adelie  
#     - Cluster 1 → Gentoo  
#     - Cluster 2 → Chinstrap  
# 
# **Conclusão:**  
# O modelo conseguiu separar adequadamente as espécies de pinguins usando apenas medições físicas.
