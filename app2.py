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
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df['combined_no_stopwords_removed'])

# Target variable
y = df['Suicidal']

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Initialize and train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Initialize LIME Explainer
explainer = LimeTextExplainer(class_names=['0', '1'])

# --- Streamlit App UI ---
st.set_page_config(page_title="Suicide Detection", layout="centered")
st.title("🧠 Suicide Ideation Detection using NLP")
st.markdown("This app detects suicidal ideation in Hinglish posts using a Logistic Regression model. It also explains predictions using LIME.")

st.markdown("---")
st.subheader("💬 Enter a sentence below:")
text_instance = st.text_area("Type your sentence here", height=150)
# Add a button to trigger the prediction
if st.button('Detect Suicidal Ideation'):
    if text_instance:
        # Vectorize input
        X_input = vectorizer.transform([text_instance])

        # Predict the class
        prediction = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]

        # Show result
        st.markdown("### 🔍 Prediction Result:")
        if prediction == 1:
            st.error("⚠️ **Suicidal ideation detected.**")
        else:
            st.success("✅ **No suicidal ideation detected.**")

        # Show probabilities
        st.markdown("### 📊 Class Probabilities")
        st.write(f"**Non-Suicidal (0):** {proba[0]:.4f} &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; **Suicidal (1):** {proba[1]:.4f}")

        # LIME explanation
        st.markdown("### 🧠 Explanation with LIME")
        st.markdown("> 🟥 **Red values** = Strong indicators of suicidal ideation  \n"
                    "> 🟦 **Lighter values** = Weak or opposite indicators")
        exp = explainer.explain_instance(
            text_instance,
            classifier_fn=lambda x: model.predict_proba(vectorizer.transform(x)),
            num_features=10
        )

        # Explanation as a list
        explanation = exp.as_list()
        exp_df = pd.DataFrame(explanation, columns=["Feature", "Importance"])
        st.dataframe(exp_df.style.background_gradient(cmap='Reds', subset=["Importance"]))

        # Optional: Visualization chart
        with st.expander("📈 Show LIME Visualization Chart"):
            fig = exp.as_pyplot_figure()
            st.pyplot(fig)