# gl_classifier.py

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


# Preprocess Data
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return " ".join(tokens)


def preprocess_data(df, text_column):
    df = df.drop_duplicates()
    df[text_column] = df[text_column].apply(preprocess_text)
    return df


# Describe Data
def describe_data(df, text_column, target_column):
    print(f"Number of entries: {len(df)}")
    print(f"Number of unique {target_column}: {df[target_column].nunique()}")
    print(
        f"Top 5 most frequent {target_column}:\n{df[target_column].value_counts().head()}"
    )
    print(f"Top 5 descriptions:\n{df[text_column].head()}")
    print(
        f"Distribution of descriptions per {target_column}:\n{df[target_column].value_counts().describe()}"
    )


# Balance Data
def balance_data(df, target_column):
    counts = df[target_column].value_counts()
    mean_count = counts.mean()
    std_count = counts.std()

    upper_bound = mean_count + 2 * std_count
    lower_bound = mean_count - 2 * std_count

    balanced_df = df[df[target_column].map(counts) < upper_bound]
    balanced_df = balanced_df[balanced_df[target_column].map(counts) > lower_bound]

    return balanced_df


# Train and Save Model
def train_and_save_model(
    train_df, text_column, target_column, model_path, method="tfidf_svm", max_len=100
):
    if method == "tfidf_svm":
        X_train = train_df[text_column]
        y_train = train_df[target_column]

        pipeline = Pipeline(
            [("tfidf", TfidfVectorizer()), ("svc", SVC(probability=True))]
        )

        param_grid = {
            "svc__C": [0.1, 1, 10],
            "svc__kernel": ["linear", "rbf"],
            "svc__gamma": ["scale", "auto"],
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1_macro")
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        joblib.dump(best_model, model_path)
        print(f"Model saved to {model_path}")

    elif method == "embedding_lstm":
        X_train = train_df[text_column]
        y_train = train_df[target_column]

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train)
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)

        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_train_categorical = to_categorical(y_train_encoded)

        vocab_size = len(tokenizer.word_index) + 1

        model = Sequential(
            [
                Embedding(vocab_size, 128, input_length=max_len),
                SpatialDropout1D(0.2),
                LSTM(100, dropout=0.2, recurrent_dropout=0.2),
                Dense(y_train_categorical.shape[1], activation="softmax"),
            ]
        )

        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        model.fit(
            X_train_pad,
            y_train_categorical,
            epochs=5,
            batch_size=64,
            validation_split=0.2,
            verbose=2,
        )
        prefix = model_path.split("_model")[0]
        model.save(model_path)
        joblib.dump(
            (tokenizer, label_encoder), prefix + "tokenizer_label_encoder.joblib"
        )
        print(
            f"Model and tokenizers saved to {model_path} and tokenizer_label_encoder.joblib"
        )


# Load Model and Evaluate
def load_and_evaluate_model(
    model_path, test_df, text_column, target_column, method="tfidf_svm", max_len=100
):
    if method == "tfidf_svm":
        best_model = joblib.load(model_path)
        X_test = test_df[text_column]
        y_test = test_df[target_column]

        y_pred = best_model.predict(X_test)

        f1 = f1_score(y_test, y_pred, average="macro")
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")

    elif method == "embedding_lstm":
        model = load_model(model_path)
        prefix = model_path.split("_model")[0]
        tokenizer, label_encoder = joblib.load(
            prefix + "tokenizer_label_encoder.joblib"
        )

        X_test = test_df[text_column]
        y_test = test_df[target_column]

        X_test_seq = tokenizer.texts_to_sequences(X_test)
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

        y_test_encoded = label_encoder.transform(y_test)
        y_test_categorical = to_categorical(y_test_encoded)

        loss, accuracy = model.evaluate(X_test_pad, y_test_categorical, verbose=0)
        y_pred = model.predict(X_test_pad)
        y_pred_classes = np.argmax(y_pred, axis=1)

        f1 = f1_score(y_test_encoded, y_pred_classes, average="macro")

        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")


# Predict Function
def predict_gl_code(model_path, description, method="tfidf_svm", max_len=100):
    processed_description = preprocess_text(description)

    if method == "tfidf_svm":
        best_model = joblib.load(model_path)
        prediction = best_model.predict([processed_description])[0]
        confidence = np.max(best_model.predict_proba([processed_description]))
        return prediction, confidence

    elif method == "embedding_lstm":
        model = load_model(model_path)
        prefix = model_path.split("_model")[0]
        tokenizer, label_encoder = joblib.load(
            prefix + "tokenizer_label_encoder.joblib"
        )

        seq = tokenizer.texts_to_sequences([processed_description])
        pad_seq = pad_sequences(seq, maxlen=max_len)

        prediction = model.predict(pad_seq)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)

        return label_encoder.inverse_transform([predicted_class])[0], confidence.item()
