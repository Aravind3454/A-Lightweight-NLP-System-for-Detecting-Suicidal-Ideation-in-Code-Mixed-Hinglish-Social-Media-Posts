import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from lime.lime_text import LimeTextExplainer
import pandas as pd

# Assuming df is already preloaded and preprocessed as you mentioned

df_uncleaned = pd.read_csv('E:/Third Year/NLP and Text Analytics/Project Docs/hhinglish_suicidal_postss.csv')

df = df_uncleaned.drop(['Upvotes','Comments','URL'], axis=1)

# %pip install unidecode

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)                      # Remove special characters
    text = re.sub(r'\s+', ' ', text)                     # Remove extra spaces
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)


from unidecode import unidecode

nltk.download('punkt')
nltk.download('stopwords')

# Expanded slang dictionary
abbrev_dict = {
    "btw": "by the way",
    "imo": "in my opinion",
    "idk": "i do not know",
    "tbh": "to be honest",
    "lol": "laughing out loud",
    "brb": "be right back",
    "omg": "oh my god",
    "wanna": "want to",
    "gonna": "going to",
    "gotta": "got to",
    "ain't": "is not",
    "ya": "you",
    "cuz": "because",
    "lemme": "let me",
    "dunno": "do not know",
    "smh": "shaking my head",
    "ikr": "i know right",
    "fr": "for real",
    "sus": "suspicious",
    "raw dawg": "unprotected",
    "raw dawg-ing": "engaging without protection",
    "don’t": "do not",
    "can't": "cannot",
    "won’t": "will not",
    "ain’t": "is not",
    "u": "you",
    "ur": "your",
    "thx": "thanks",
    "plz": "please",
    "b4": "before",
    "nvm": "never mind"
}

# Function to expand abbreviations
def expand_abbreviations(text, abbrev_dict):
    for abbr, full in abbrev_dict.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text)
    return text
# Function to clean text with optional stopword removal
def clean_text(text, remove_stopwords=True):
    text = str(text).lower()
    text = unidecode(text)  # Fix misencoded characters
    text = expand_abbreviations(text, abbrev_dict)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)                      # Remove special characters
    text = re.sub(r'\s+', ' ', text)                     # Normalize whitespace
    tokens = nltk.word_tokenize(text)
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


# Preprocess each part separately
df['clean_title'] = df['Title'].fillna('').apply(lambda x: clean_text(x, remove_stopwords=True))
df['clean_text'] = df['Text'].fillna('').apply(lambda x: clean_text(x, remove_stopwords=True))

# Combine cleaned title and text
df['processed_text'] = df['clean_title'] + ' ' + df['clean_text']

# Now create version without stopword removal
df['title_no_stopwords_removed'] = df['Title'].fillna('').apply(lambda x: clean_text(x, remove_stopwords=False))
df['text_no_stopwords_removed'] = df['Text'].fillna('').apply(lambda x: clean_text(x, remove_stopwords=False))

# Combined version without stopword removal
df['combined_no_stopwords_removed'] = df['title_no_stopwords_removed'] + ' ' + df['text_no_stopwords_removed']
# Combine relevant text fields
df['combined'] = df['Title'].fillna('') + ' ' + df['Text'].fillna('')

import nltk
from nltk.tokenize import word_tokenize

# Tokenization
df['tokens'] = df['combined_no_stopwords_removed'].apply(word_tokenize)



# Apply cleaning
df['processed_text'] = df['combined'].apply(clean_text)
# Initialize TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF Vectorizer with N-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # unigrams + bigrams
# OR  vectorizer = TfidfVectorizer(ngram_range=(1, 3))  # unigrams + bigrams + trigrams

# Fit and transform
X = vectorizer.fit_transform(df['combined_no_stopwords_removed'])  # Assuming your text column is 'text'
y = df['Suicidal']  # Your target column (0 or 1)

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Initialize TF-IDF (if not already done)
# vectorizer = TfidfVectorizer(ngram_range=(1, 2))
# X = vectorizer.fit_transform(df['combined_no_stopwords_removed'])
# y = df['Suicidal']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# # Evaluation
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

from lime.lime_text import LimeTextExplainer


# --- Streamlit App UI ---
st.set_page_config(page_title="Suicide Detection", layout="centered")
st.title("🧠 Suicide Ideation Detection using NLP")
st.markdown("This app detects suicidal ideation in Hinglish posts using a XGBoost model. It also explains predictions using LIME.")

st.markdown("---")
st.subheader("💬 Enter a sentence below:")
text_instance = st.text_area("Type your sentence here", height=150)
if st.button('Detect using XGBoost'):
    if text_instance:
        # Prediction
        prediction_xgb = xgb_model.predict(vectorizer.transform([text_instance]))[0]

        st.markdown("### 🔍 XGBoost Prediction Result:")
        if prediction_xgb == 1:
            st.write("⚠️ **Suicidal ideation detected.**")
        else:
            st.write("✅ **No suicidal ideation detected.**")

        # LIME explanation
        explainer_xgb = LimeTextExplainer(class_names=['0', '1'])

        exp_xgb = explainer_xgb.explain_instance(
            text_instance,
            classifier_fn=lambda x: xgb_model.predict_proba(vectorizer.transform(x)),
            num_features=10
        )

        st.write("### Explanation of Prediction (XGBoost + LIME)")
        explanation = exp_xgb.as_list()
        for feature, importance in explanation:
            st.write(f"{feature}: {importance:.4f}")

        # Bar chart explanation
        import matplotlib.pyplot as plt
        features = [item[0] for item in explanation]
        importances = [item[1] for item in explanation]
        colors = ['red' if val > 0 else 'blue' for val in importances]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(features, importances, color=colors)
        ax.set_title("🔍 LIME Explanation (XGBoost)")
        ax.set_xlabel("Contribution to Prediction")
        ax.invert_yaxis()

        st.pyplot(fig)
