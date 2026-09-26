import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.manifold import TSNE

data = pd.read_csv('health_fitness_dataset.csv')

intensity_map = {'Low': 0, 'Medium': 1, 'High': 2}
data['intensity_numeric'] = data['intensity'].map(intensity_map)

activity_dummies = pd.get_dummies(data['activity_type'], prefix='activity')

consistency = data.groupby('participant_id').size().rename('consistency')

combined = pd.concat([
    data[['participant_id', 'intensity_numeric']],
    activity_dummies
], axis=1)

grouped = combined.groupby('participant_id').mean().reset_index()
grouped = grouped.merge(consistency, on='participant_id')

features = grouped[[
    'intensity_numeric',
    'consistency',
    *[col for col in grouped.columns if col.startswith('activity_')]
]]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

best_k = 3
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
final_labels = kmeans.fit_predict(scaled_features)

grouped['cluster'] = final_labels
grouped['silhouette_score'] = silhouette_samples(scaled_features, final_labels)

cluster_order = grouped.groupby('cluster')['intensity_numeric'].mean().sort_values()
intensity_cluster_map = {
    cluster_order.index[0]: 'Low',
    cluster_order.index[1]: 'Medium',
    cluster_order.index[2]: 'High'
} if best_k == 3 else None

if intensity_cluster_map:
    grouped['intensity_cluster'] = grouped['cluster'].map(intensity_cluster_map)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(scaled_features)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1],
                hue=grouped['cluster'], palette='Set2', s=80)
plt.title(f'Users clustered based on workout intensity, k={best_k}')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(title='Cluster')
plt.show()

