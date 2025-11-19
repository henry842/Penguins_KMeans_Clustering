# Penguins K-Means Clustering

## Descrição
Este projeto explora o uso do algoritmo de clustering **K-Means** para segmentar espécies de pinguins com base em características físicas (comprimento e profundidade do bico, comprimento da barbatana e massa corporal). O objetivo é demonstrar como K-Means pode ser aplicado em dados biológicos e interpretar os clusters em relação às espécies reais.

## Dataset
O dataset utilizado é o `penguins` do pacote `seaborn`, que contém medições físicas de três espécies de pinguins: Adelie, Chinstrap e Gentoo.

## Etapas do Projeto
1. Limpeza dos dados e remoção de valores ausentes.
2. Seleção de variáveis numéricas.
3. Visualizações exploratórias (pairplot, heatmap, boxplots).
4. Padronização dos dados.
5. Aplicação do K-Means com 3 clusters.
6. Visualização dos clusters e centroides.
7. Avaliação da qualidade dos clusters com Silhouette Score e Davies-Bouldin Score.
8. Redução de dimensionalidade com PCA para visualização 2D.
9. Comparação dos clusters com as espécies reais.
10. Análise final e interpretação dos clusters.

## Resultados
- **Número de clusters:** 3  
- **Silhouette Score:** 0.446  
- **Davies-Bouldin Score:** 0.942  
- **Correspondência com espécies reais:**
  - Cluster 0 → Adelie
  - Cluster 1 → Gentoo
  - Cluster 2 → Chinstrap

O modelo conseguiu separar adequadamente as espécies usando apenas medições físicas.

## Tecnologias
- Python 3
- Pandas
- Seaborn
- Plotly
- Scikit-learn
- Matplotlib
