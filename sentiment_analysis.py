import sys
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

sia = SentimentIntensityAnalyzer()
stop_words = stopwords.words("english")


def analyze_sentiment(comment):
    score = sia.polarity_scores(comment)
    if score["compound"] >= 0.05:
        return "Positive"
    elif score["compound"] <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def extract_insights(comment):
    if len(comment.split()) < 2:
        return ""
    vectorizer = CountVectorizer(stop_words=stop_words)
    word_counts = vectorizer.fit_transform([comment])
    if not vectorizer.vocabulary_ or len(vectorizer.vocabulary_) == 0:
        return ""
    words = vectorizer.get_feature_names_out()
    total_counts = word_counts.toarray().sum(axis=0)
    word_freq = {word: count for word, count in zip(words, total_counts)}
    filtered_word_freq = {
        word: count for word, count in word_freq.items() if word not in stop_words
    }
    if not filtered_word_freq or len(filtered_word_freq) == 0:
        return ""
    sorted_words = sorted(filtered_word_freq.items(), key=lambda x: x[1], reverse=True)
    insights = [word for word, freq in sorted_words[:5]]
    return ", ".join(insights)


def analyze(input_file, output_file="sentiment.csv"):
    df = pd.read_csv(input_file)
    print(df.describe())
    df["Sentiment"] = df["Comments"].apply(analyze_sentiment)
    df["Actionable Insights"] = df["Comments"].apply(lambda x: extract_insights(str(x)))
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sentiment_analysis.py <input_file> <output_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    analyze(input_file, output_file)
