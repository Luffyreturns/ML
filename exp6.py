# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns

data = pd.read_csv(r'C:\Users\omkar\OneDrive\Desktop\ML\dataset\Wholesale customers data.csv')
print("Data Overview:\n", data.head())

data_scaled = normalize(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
print("Normalized Data:\n", data_scaled.head())

plt.figure(figsize=(10, 7))
plt.title("Dendrogram for Agglomerative Clustering")
dendrogram = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.show()

cluster = AgglomerativeClustering(n_clusters=2, linkage='ward')
data['Cluster'] = cluster.fit_predict(data_scaled)

# Plot clusters for every feature against each other using pairplot
features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']  # Correct column name

# Create pairplot to visualize clustering
sns.pairplot(data, vars=features, hue='Cluster', palette='Set1', diag_kind='kde')
plt.suptitle('Pairplot of Clusters Across All Features', y=1.02)
plt.show()

