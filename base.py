################
# BASE: sviluppo di un sistema di raccomandazione basato su collaborative filtering
# partendo da un set di dati di review di prodotti Amazon.
################
# Ok, si puo estendere 1. Analisi Esplorativa (statistiche descrittive, analisi correlazione)
# Ok, l'algoritmo è buggato 2. Identificazione della configurazione ottimale dell’algoritmo K-NN per la predizione dei rating.
# In questo punto dovranno quindi essere testate le diverse combinazioni: similarità, valore di K, user/item based.
# Tramite le diverse metriche di performance (MSE e RMSE) individuare di conseguenza la configurazione ottimale.
# Ok, dispendioso in termini di memoria, riempio matrice ridotta 3. Filling della matrice di rating con la configurazione ottimale
# Ok, magari trovare un numero di cluster migliore 4. Segmentazione degli utenti in base alle preferenze: algoritmo di clustering K-MEANS con cosine similarity.
# Ok, dispendioso in termini di memoria, riempio solo per uno user 5. Creazione per ogni utente della lista degli n items (top k items) da consigliare (es. considerando il rating predetto).
# Ok 6. Filling della matrice di rating attraverso Matrix Factorization in aggiunta a K-NN e confronto dei risultati ottenuti in termini di MSE e RMSE.
# Librerie suggerite: Surprise e Scikit-Learn
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

from lib import knn, matrix_factorization, common

RANDOM_STATE = 1

sns.set_theme(context="paper", style="darkgrid")

sns.set_palette("Spectral")
# sns.set_palette("viridis")


################################################################
# Preparazione dataset
################################################################

# df = common.prepare_ratings_dataframe('Industrial_and_Scientific', RANDOM_STATE, min_review_per_item=8, min_review_per_user=6)
# df = common.prepare_ratings_dataframe('Subscription_Boxes', RANDOM_STATE, min_review_per_item=0, min_review_per_user=0)

# df = common.prepare_ratings_dataframe('Tools_and_Home_Improvement', RANDOM_STATE, min_review_per_item=14, min_review_per_user=14)
# df = common.prepare_ratings_dataframe('Toys_and_Games', RANDOM_STATE, min_review_per_item=13, min_review_per_user=10)
df = common.prepare_ratings_dataframe("Automotive", RANDOM_STATE, min_review_per_item=14, min_review_per_user=13)

print("Dataset:")
print(df.head())

print("Nº Utenti", df["user_id"].nunique())
print("Nº Prodotti", df["parent_asin"].nunique())
print("Nº Rating", df["rating"].count())

srp_dataset = common.prepare_reviews_dataset(df)


################################################################
# Individuo configurazione ottimale K-NN tramite GridSearch
################################################################

# gs = knn.hyperparam_opt(srp_dataset)
# opt = common.get_opt_values(gs, True)

# Migliori parametri trovati tramite GridSearch:
# {'k': 10, 'sim_options': {'name': 'cosine', 'user_based': True}}
# Miglior RMSE score ottenuto:
# 1.0375800709090214


################################################################
# Modello di raccomandazione tramite K-NN
################################################################

(knn_model, knn_mse, knn_rmse) = knn.prepare_model(srp_dataset, RANDOM_STATE)

common.test_prediction(df, knn_model)

print("Building rating matrix...")
rating_matrix_knn = common.build_rating_matrix(
    df[:1000], knn_model, remove_rated=True
)  # dimensioni ridotte per velocizzare
print("Matrice di rating tramite K-NN (25000 rows)")
print(rating_matrix_knn.head())
print("Predizioni impossibili:", common.debug_count)
print(
    f"Percentuale predizioni impossibili su predizioni calcolate:{common.debug_count * 100 / common.debug_pred_count:.02f}%"
)
common.debug_count = 0
common.debug_pred_count = 0


################################################################
# Segmentazione tramite K-Means con cosine similarity
################################################################

kmeans_df = df.groupby("user_id").agg(
    mean_rating=("rating", "mean"), count_rating=("rating", "size")
)

print("Tabella di dati per K-Means")
print(kmeans_df.head())

# Applicazione dell'algoritmo K-Means
num_clusters = 5
kmeans = KMeans(init="k-means++", n_clusters=num_clusters, random_state=RANDOM_STATE)
user_clusters = kmeans.fit_predict(kmeans_df[["mean_rating", "count_rating"]])

# Aggiunta dei cluster al DataFrame originale
kmeans_df["cluster"] = user_clusters

# Visualizzazione delle dimensioni dei cluster
cluster_counts = kmeans_df["cluster"].value_counts()

print("Dimensioni dei cluster:", cluster_counts)

# Plot dei count dei cluster
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.countplot(x="cluster", data=kmeans_df)
plt.title("Distribuzione degli Utenti nei Cluster")
plt.xlabel("Cluster")
plt.ylabel("Numero di Utenti")
plt.yscale("log")

plt.subplot(1, 2, 2)
sns.scatterplot(
    x="count_rating",
    y="mean_rating",
    legend="full",
    c=user_clusters,
    cmap="viridis",
    alpha=0.6,
    data=kmeans_df[["mean_rating", "count_rating"]],
    s=30,
)
plt.scatter(
    kmeans.cluster_centers_[:, 1],
    kmeans.cluster_centers_[:, 0],
    s=40,
    c="black",
    label="centroid",
)
plt.title("Distribuzione degli Utenti nei Cluster")
plt.ylabel("Media dei rating per utente")
plt.xlabel("Numero di recensioni per utente")

plt.tight_layout()
plt.show(block=False)


################################################################
# Individuo configurazione ottimale Matrix Factorization tramite GridSearch
################################################################

# gs = matrix_factorization.hyperparam_opt(srp_dataset)
# opt = common.get_opt_values(gs, True)
# # Migliori parametri trovati tramite GridSearch:
# {'n_factors': 100}
# # Miglior RMSE score ottenuto:
# 0.8875611375991975


################################################################
# Matrix Factorization (SVD)
################################################################

(svd_model, svd_mse, svd_rmse) = matrix_factorization.prepare_model(
    srp_dataset, RANDOM_STATE
)


################################################################
# Utente/Item/Rating di verifica per SVD
################################################################

svd_pred = common.test_prediction(df, svd_model)

svd_recomms = common.build_top_k_recommendation_for_user(
    svd_pred.uid, df, svd_model, remove_rated=True
)
knn_recomms = common.build_top_k_recommendation_for_user(
    svd_pred.uid, df, knn_model, remove_rated=True
)

print("Raccomandazioni tramite K-NN:", svd_recomms)
print("Raccomandazioni tramite Matrix Factorization:", knn_recomms)


################################################################
# Filling della matrice di rating tramite Matrix Factorization(SVD)
################################################################

print("Building rating matrix with SVD...")
rating_matrix_svd = common.build_rating_matrix(
    df[0:1000], svd_model, remove_rated=True
)  # dimensioni ridotte per velocizzare
print("Matrice di rating tramite Matrix Factorization (25000 rows)")
print(rating_matrix_svd.head())


################################################################
# Confronto metriche
################################################################

# Confronto delle metriche (MSE e RMSE) per K-NN e SVD
metrics_comparison = pd.DataFrame(
    {
        "Algorithm": ["K-NN", "SVD"],
        "MSE": [knn_mse, svd_mse],
        "RMSE": [knn_rmse, svd_rmse],
    }
)

print("Confronto delle metriche:", metrics_comparison)

# Plot del confronto delle metriche
plt.figure(figsize=(10, 6))

# MSE plot
plt.subplot(1, 2, 1)
sns.barplot(x="Algorithm", y="MSE", data=metrics_comparison)
plt.title("Confronto MSE")
plt.ylabel("MSE")
plt.xlabel("Algoritmo")

# RMSE plot
plt.subplot(1, 2, 2)
sns.barplot(x="Algorithm", y="RMSE", data=metrics_comparison)
plt.title("Confronto RMSE")
plt.ylabel("RMSE")
plt.xlabel("Algoritmo")

plt.tight_layout()
plt.show(block=False)

input("Premi invio per uscire...")
