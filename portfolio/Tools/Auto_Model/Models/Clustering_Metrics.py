from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class ClusteringMetrics:
    def evaluate_and_print(self, model, X, labels, hyperparams):
        if len(set(labels)) < 2:
            print(f"Hyperparameters: {hyperparams}")
            print("Cannot compute metrics: less than 2 clusters")
            print("-" * 50)
            return -1
        
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies = davies_bouldin_score(X, labels)

        print(f"Hyperparameters: {hyperparams}")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Calinski-Harabasz Score: {calinski:.4f}")
        print(f"Davies-Bouldin Score: {davies:.4f}")
        print("-" * 50)

        return silhouette
