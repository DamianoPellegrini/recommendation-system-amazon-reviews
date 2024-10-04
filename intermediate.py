################
# INTERMEDIO: sviluppo di un sistema di raccomandazione collaborative filtering
# (progetto base) e content based, partendo da un set di dati di review di prodotti Amazon.
################
# 1. Processamento degli attributi testuali dei diversi prodotti (almeno i campi title e description) con le
# tecniche di Natural Language Processing viste in laboratorio.
# 2. Embedding dei campi con una tecnica basata sulla frequenza (bag-of-words o TFIDF) e una tecnica
# neurale (transformers).
# 3. Effettuare la predizione dei rating attraverso l’algoritmo K-NN per ogni utente usando gli embedding
# ottenuti con le due tecniche del punto 2.
# 4. Valutazione critica dei risultati ottenuti con le due diverse tecniche di embedding
# 5. Valutazione critica dei risultati ottenuti con il sistema di raccomandazione collaborative filtering
# (progetto base) e quello content-based (punto 1-3)
# Librerie suggerite Scikit-Learn, NLTK, Hugging-face

import datasets
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import lib
import lib.common

nltk.download("punkt")  # Tokenizer
nltk.download("stopwords")

# reviews = datasets.load_dataset(
#     "McAuley-Lab/Amazon-Reviews-2023",
#     "0core_rating_only_Automotive",
#     split="full",
#     trust_remote_code=True,
# )
# df_rev = reviews.to_pandas()

# df_rev = base.common.filter_dataframe(
#     df_rev, min_review_per_item=14, min_review_per_user=13
# )

df_rev = lib.common.prepare_ratings_dataframe(
    'Automotive', random_state=1337, min_review_per_item=14, min_review_per_user=13
)

print("N. Users:", len(df_rev["user_id"].unique()))
print("N. Items:", len(df_rev["parent_asin"].unique()))
print("N. Ratings:", len(df_rev))

# Prepare meta dataset
meta = datasets.load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_meta_Automotive",
    split="full",
    trust_remote_code=True,
)
df_meta = meta.to_pandas()

# Preprocess some fields
# df_rev['has_images'] = df_rev['images'].apply(lambda x: len(x) > 0)
# df_rev = df_rev.drop(columns=["text", "title", "images"])
df_meta["has_images"] = df_meta["images"].apply(lambda x: len(x) > 0)
df_meta["has_videos"] = df_meta["videos"].apply(lambda x: len(x) > 0)
# Drop unused fields
df_meta = df_meta.drop(
    columns=["images", "videos", "bought_together", "subtitle", "author"]
)

# drop any review whose item has no description or title, after converting description to a string
df_meta["description"] = df_meta["description"].apply(
    lambda x: " ".join(x) if isinstance(x, list) or isinstance(x, np.ndarray) else x
)
df_meta = df_meta.dropna(subset=["title", "description"])

# merge meta and reviews
df = pd.merge(df_rev, df_meta, on="parent_asin")
print(df.columns)
print(df.head())


# NLP Preprocessing
stemmer = nltk.PorterStemmer()


def stemming_tokenizer(text: str) -> list:
    tokens = nltk.word_tokenize(text)
    tokens = [
        stemmer.stem(word) for word in tokens if word.isascii() and word.isalnum()
    ]
    return tokens


df["corpus_text"] = df["title"] + " " + df["description"]
print(df[["corpus_text"]].head())

#! EMBEDDING
# Embedding con BoW
# vectorizer = CountVectorizer(max_features=5000, stop_words=["english"], tokenizer=stemming_tokenizer)
# vectorizer = CountVectorizer(max_features=5000, stop_words=["english"])
vectorizer = CountVectorizer(stop_words=["english"], tokenizer=stemming_tokenizer)
# vectorizer = CountVectorizer(stop_words=["english"])
bow_embeddings = vectorizer.fit_transform(df["corpus_text"])
bow_dataset = pd.DataFrame(
    bow_embeddings.toarray(), columns=vectorizer.get_feature_names_out()
)
bow_dataset["parent_asin"] = df["parent_asin"]
print(bow_dataset.head())

# Embedding con Transformers
model = SentenceTransformer("sentence-transformers/average_word_embeddings_komninos")
trans_embeddings = model.encode(
    df["corpus_text"], show_progress_bar=True, batch_size=128
)
trans_dataset = pd.DataFrame(trans_embeddings)
trans_dataset["parent_asin"] = df["parent_asin"]


# Funzione per addestrare il modello per ogni utente
def train_user_model(user_id: str, df: pd.DataFrame, embeddings: pd.DataFrame, mse_users: list):
    try:
        user_reviews = df[df["user_id"] == user_id]
        rated_items_embedding = embeddings[
            embeddings["parent_asin"].isin(user_reviews["parent_asin"])
        ]
        dataset_rec = pd.merge(user_reviews, rated_items_embedding, on="parent_asin")
        dataset_rec = dataset_rec.drop(columns=["parent_asin", "user_id"])
        column_name = (
            "rating_x" if "rating" in rated_items_embedding.columns else "rating"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            dataset_rec.drop(columns=column_name),
            dataset_rec[column_name],
            test_size=0.20,
            random_state=0,
        )

        neigh_reg = KNeighborsRegressor(
            n_neighbors=min(40, len(X_train)), metric="cosine"
        )
        neigh_reg.fit(X_train, y_train)
        y_pred = neigh_reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_users.append(mse)
    except Exception as e:
        print(f"Error processing user {user_id}: {e}")
        raise e
    
def train_user_model_batch(user_id_batch: np.ndarray, df: pd.DataFrame, embeddings: pd.DataFrame, mse_users: list):
    for user_id in user_id_batch:
        try:
            user_reviews = df[df["user_id"] == user_id]
            rated_items_embedding = embeddings[
                embeddings["parent_asin"].isin(user_reviews["parent_asin"])
            ]
            dataset_rec = pd.merge(user_reviews, rated_items_embedding, on="parent_asin")
            dataset_rec = dataset_rec.drop(columns=["parent_asin", "user_id"])
            column_name = (
                "rating_x" if "rating" in rated_items_embedding.columns else "rating"
            )

            X_train, X_test, y_train, y_test = train_test_split(
                dataset_rec.drop(columns=column_name),
                dataset_rec[column_name],
                test_size=0.20,
                random_state=0,
            )

            neigh_reg = KNeighborsRegressor(
                n_neighbors=min(40, len(X_train)), metric="cosine"
            )
            neigh_reg.fit(X_train, y_train)
            y_pred = neigh_reg.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mse_users.append(mse)
        except Exception as e:
            print(f"Error processing user {user_id}: {e}")
            raise e


# Parallelizzazione dell'addestramento
bow_mse_users = []
unique_user_ids = df["user_id"].unique()

print(f"Inizio KNN BoW")
# Parallel(n_jobs=-1, backend="threading")(
#     delayed(train_user_model)(user_id, df[['user_id', 'parent_asin', 'rating']], bow_dataset, bow_mse_users)
#     for user_id in unique_user_ids
# )
batch_size = 128  # Dimensione del batch di utenti
Parallel(n_jobs=-1, backend="threading")(
    delayed(train_user_model_batch)(unique_user_ids[i:i+batch_size], df[['user_id', 'parent_asin', 'rating']], bow_dataset, bow_mse_users)
    for i in range(0, len(unique_user_ids), batch_size)
)

print(f"Average MSE across all users: {np.mean(bow_mse_users):.8f}")
print(f"Average RMSE across all users: {np.sqrt(np.mean(bow_mse_users)):.8f}")
# Average MSE across all users: 0.07828542
# Average RMSE across all users: 0.27979532

trans_mse_users = []
print(f"Inizio KNN Transformer")
# Parallel(n_jobs=-1, backend="threading")(
#     delayed(train_user_model)(user_id, df[['user_id', 'parent_asin', 'rating']], trans_dataset, trans_mse_users)
#     for user_id in unique_user_ids
# )
batch_size = 128  # Dimensione del batch di utenti
Parallel(n_jobs=-1, backend="threading")(
    delayed(train_user_model_batch)(unique_user_ids[i:i+batch_size], df[['user_id', 'parent_asin', 'rating']], trans_dataset, trans_mse_users)
    for i in range(0, len(unique_user_ids), batch_size)
)

print(f"Average MSE across all users: {np.mean(trans_mse_users):.8f}")
print(f"Average RMSE across all users: {np.sqrt(np.mean(trans_mse_users)):.8f}")
# Average MSE across all users: 0.07966070
# Average RMSE across all users: 0.28224227

input("Premi per terminare...")
exit()
