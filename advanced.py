################
# AVANZATO: sviluppare un sistema di raccomandazione collaborative filtering (progetto base), content based
# (progetto intermedio) ed effettuare una sentiment analysis sulle recensioni dei prodotti.
################
# 1. Processamento degli attributi testuali relativi alle review dei diversi prodotti (almeno i campi title e text
# del file user reviews) con tecniche di Natural Language Processing.
# 2. Embedding dei campi con una tecnica basata sulla frequenza (bag-of-words o TFIDF) e una tecnica
# neurale (transformers).
# 3. Effettuare la predizione del sentiment (rating 1-2: sentiment negativo, rating 3: sentiment neutro, rating
# 4-5: sentiment positivo) utilizzando gli algoritmi di classificazione di Scikit-Learn (quelli visti in
# laboratorio).

import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import lib.common

print("Loading dataset...")
df = lib.common.prepare_reviews_dataframe('Automotive', min_review_per_item=15, min_review_per_user=13)
print("Dataset loaded.")

print(df.describe())

def to_sentiment(rating):
    rating = int(rating)

    if np.isnan(rating):
        breakpoint()

    if rating <= 2:
        return np.float32(0)
    elif rating == 3:
        return np.float32(1)
    else:
        return np.float32(2)

SENTIMENTS = ['negativo', 'neutrale', 'positivo']

print("Preprocessing...")
df["text"] = df["text"].apply(
    lambda x: " ".join(x) if isinstance(x, list) or isinstance(x, np.ndarray) else x
)
df = df.dropna(subset=["title", "text"])

df['sentiment'] = df['rating'].dropna().apply(to_sentiment)
df["corpus_text"] = df["title"] + ". " + df["text"]

print(df)

nltk.download("punkt")  # Tokenizer
stemmer = nltk.PorterStemmer()

def stemming_tokenizer(text: str) -> list:
    tokens = nltk.word_tokenize(text)
    tokens = [
        stemmer.stem(word) for word in tokens if word.isascii() and word.isalnum()
    ]
    return tokens


print("Embedding...")
#! EMBEDDING
# Embedding con BoW
vectorizer = CountVectorizer(max_features=2000, stop_words=["english"], tokenizer=stemming_tokenizer)
# vectorizer = CountVectorizer(stop_words=["english"])
bow_embeddings = vectorizer.fit_transform(df["corpus_text"])
bow_dataset = pd.DataFrame(
    bow_embeddings.toarray(), columns=vectorizer.get_feature_names_out()
)
print(bow_dataset.head())

# Embedding con Transformers
model = SentenceTransformer("sentence-transformers/average_word_embeddings_komninos")
trans_embeddings = model.encode(
    df["corpus_text"].to_list(), show_progress_bar=True, batch_size=128
)
trans_dataset = pd.DataFrame(trans_embeddings)
print(trans_dataset.head())


# dati embedding di title + review (X) predire il sentiment (Y)
def train_and_measure_model(dataset: pd.DataFrame, embeddings: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        dataset['sentiment'],
        test_size=0.20,
        random_state=0,
    )
    neigh_reg = KNeighborsClassifier(
        n_neighbors=min(30, len(X_train)), metric="cosine"
    )
    neigh_reg.fit(X_train, y_train)
    y_pred = neigh_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

print("Training...")
print("Inizio train BoW")
bow_mse = train_and_measure_model(df, bow_dataset)
print("BoW MSE:", bow_mse)
print("Inizio train Trans")
trans_mse = train_and_measure_model(df, trans_dataset)
print("Trans MSE:", trans_mse)

# BoW MSE: 0.24559395
# Trans MSE: 0.26195282
