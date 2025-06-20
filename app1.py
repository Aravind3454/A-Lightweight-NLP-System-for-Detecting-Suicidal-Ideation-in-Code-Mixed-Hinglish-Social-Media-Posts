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

from sklearn.svm import SVC

svm_model = SVC(kernel='linear', probability=True)  # Set probability=True


# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

# Base SVM
base_svc = LinearSVC()

# Calibration wrapper
calibrated_svc = CalibratedClassifierCV(base_svc)

# Fit on training data
calibrated_svc.fit(X_train, y_train)

y_pred = calibrated_svc.predict(X_test)

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

# Use SVC with probability=True
svc_model = SVC(kernel='linear', probability=True)

# Train
svc_model.fit(X_train, y_train)

# Create pipeline
pipeline = make_pipeline(vectorizer, svc_model)

# Initialize LIME Explainer
explainer = LimeTextExplainer(class_names=['0', '1'])

# --- Streamlit App UI ---
st.set_page_config(page_title="Suicide Detection", layout="centered")
st.title("🧠 Suicide Ideation Detection using NLP")
st.markdown("This app detects suicidal ideation in Hinglish posts using a SVC model. It also explains predictions using LIME.")

st.markdown("---")
st.subheader("💬 Enter a sentence below:")
text_instance = st.text_area("Type your sentence here", height=150)

if st.button('Detect Suicidal Ideation'):
    if text_instance:
    # Predict the class using the trained model
        prediction = svc_model.predict(vectorizer.transform([text_instance]))[0]

    # Show result
        st.markdown("### 🔍 Prediction Result:")
    # Display the predicted class
        if prediction == 1:
            st.write("⚠️ **Suicidal ideation detected.**")
        else:
            st.write("✅ **No suicidal ideation detected.**")

    # Generate LIME explanation for the entered text
        exp = explainer.explain_instance(
            text_instance,
            classifier_fn=lambda x: svc_model.predict_proba(vectorizer.transform(x)),
            num_features=10
            )

    # Show LIME explanation results
        st.write("### Explanation of Prediction")
        explanation = exp.as_list()
        for feature, importance in explanation:
            st.write(f"{feature}: {importance:.4f}")

    # Display the LIME explanation visualization (optional)
        import matplotlib.pyplot as plt

    # Extract explanation
        exp_list = exp.as_list()
        features = [item[0] for item in exp_list]
        importances = [item[1] for item in exp_list]

    # Assign colors based on importance direction
        colors = ['red' if val > 0 else 'blue' for val in importances]

    # Create horizontal bar chart
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(features, importances, color=colors)
        ax.set_title("🔍 LIME Explanation for Suicidal Ideation Prediction")
        ax.set_xlabel("Contribution to Prediction (Red = Suicidal, Blue = Non-Suicidal)")
        ax.invert_yaxis()

    # Render in Streamlit
        st.pyplot(fig)

