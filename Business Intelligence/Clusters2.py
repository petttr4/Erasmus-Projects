import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("health_fitness_dataset.csv")

data['date'] = pd.to_datetime(data['date'])

latest_entries = data.sort_values('date').groupby('participant_id').tail(1)

features = [
    'age', 'bmi', 'duration_minutes', 'calories_burned',
    'avg_heart_rate', 'hours_sleep', 'stress_level',
    'daily_steps', 'hydration_level', 'fitness_level'
]
X = latest_entries[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_optimal = 4
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
latest_entries['cluster'] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
latest_entries['PCA1'] = X_pca[:, 0]
latest_entries['PCA2'] = X_pca[:, 1]

# Scatter plot of clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=latest_entries, x='PCA1', y='PCA2', hue='cluster', palette='Set2')
plt.title('User Segmentation Based on Health & Fitness ')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

cluster_summary = latest_entries.groupby('cluster')[features].mean().round(2)
print("Cluster Summary:")
print(cluster_summary)
