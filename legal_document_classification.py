import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

st.set_page_config(
    page_title="Legal Document Classification",
    page_icon="⚖️",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background: #000000;
            color: #f5f5f5;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: #ffffff;
        }
        p, li, div, span {
            color: #f5f5f5;
        }
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            color: #ffffff;
        }
        [data-testid="stSidebar"] {
            background: #111111;
            color: #ffffff;
        }
        .css-1d391kg {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 1rem;
        }
        .stDataFrame, .stTable {
            background: #111111;
            color: #ffffff;
        }
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div {
            background: #111111;
            color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚖️ Legal Document Category Classification")
st.subheader("TF-IDF + Linear SVM + Logistic Regression + Naive Bayes")

st.write(
    "This application classifies legal documents into their respective categories using "
    "natural language processing and machine learning."
)

st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload Legal Documents CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    try:
        df = pd.read_csv("legal_documents_240_mixed.csv")
    except FileNotFoundError:
        st.warning("Please upload legal_documents_240_mixed.csv")
        st.info("CSV must contain 'text' and 'category' columns.")
        st.stop()

if df.columns.duplicated().any():
    unique_columns = []
    column_counts = {}
    for column in df.columns:
        count = column_counts.get(column, 0)
        unique_columns.append(column if count == 0 else f"{column}.{count}")
        column_counts[column] = count + 1
    df.columns = unique_columns

st.header("📊 Dataset Overview")
st.write(f"Dataset shape: {df.shape}")
st.dataframe(df.head(10), use_container_width=True)

st.header("📌 Dataset Columns")
col1, col2 = st.columns(2)
with col1:
    text_column = st.selectbox("Text Column", df.columns)
with col2:
    target_column = st.selectbox(
        "Category Column",
        [column for column in df.columns if column != text_column],
    )

data = df[[text_column, target_column]].copy()
data = data.dropna()
data[text_column] = data[text_column].astype(str).str.strip()
data = data[data[text_column] != ""]

if data.empty:
    st.error("The selected text column has no usable data after cleaning.")
    st.stop()

if data[target_column].nunique() < 2:
    st.error("The selected target column must contain at least two categories.")
    st.stop()

st.header("📂 Legal Document Categories")
category_counts = data[target_column].value_counts().sort_index()
st.dataframe(category_counts.rename("Number of Documents"), use_container_width=True)

fig, ax = plt.subplots(figsize=(12, 5))
category_counts.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", n_colors=len(category_counts)))
ax.set_title("Legal Document Category Distribution")
ax.set_xlabel("Category")
ax.set_ylabel("Number of Documents")
ax.grid(axis="y", linestyle="--", alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

X = data[text_column]
y = data[target_column]

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
        stratify=y,
    )
except ValueError:
    st.error("The dataset does not contain enough samples in every category for stratified splitting.")
    st.stop()

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=20000,
    ngram_range=(1, 2),
    sublinear_tf=True,
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

st.header("🔤 Training Data Summary")
col1, col2 = st.columns(2)
with col1:
    st.metric("Training Documents", len(X_train))
with col2:
    st.metric("Testing Documents", len(X_test))

with st.spinner("Training machine learning models with the full cleaned dataset..."):
    svm_model = LinearSVC()
    svm_model.fit(X_train_tfidf, y_train)
    svm_pred = svm_model.predict(X_test_tfidf)

    logistic_model = LogisticRegression(max_iter=4000)
    logistic_model.fit(X_train_tfidf, y_train)
    logistic_pred = logistic_model.predict(X_test_tfidf)

    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    nb_pred = nb_model.predict(X_test_tfidf)

svm_accuracy = accuracy_score(y_test, svm_pred)
logistic_accuracy = accuracy_score(y_test, logistic_pred)
nb_accuracy = accuracy_score(y_test, nb_pred)

st.header("📈 Model Performance")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Linear SVM", f"{svm_accuracy * 100:.2f}%")
with col2:
    st.metric("Logistic Regression", f"{logistic_accuracy * 100:.2f}%")
with col3:
    st.metric("Naive Bayes", f"{nb_accuracy * 100:.2f}%")

results = pd.DataFrame(
    {
        "Model": ["Linear SVM", "Logistic Regression", "Naive Bayes"],
        "Accuracy": [svm_accuracy * 100, logistic_accuracy * 100, nb_accuracy * 100],
    }
)

st.subheader("📊 Accuracy Comparison")

colors = ["#1f77b4", "#2ca02c", "#d62728"]
fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
ax_bar = sns.barplot(data=results, x="Model", y="Accuracy", palette=colors, ax=ax_bar)
for container in ax_bar.containers:
    ax_bar.bar_label(container, fmt="%.2f%%", padding=5, fontsize=11, fontweight="bold")
ax_bar.set_title("Legal Document Classification - Model Accuracy Comparison", fontsize=15, fontweight="bold", pad=15)
ax_bar.set_xlabel("Machine Learning Model", fontsize=12, fontweight="bold")
ax_bar.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
ax_bar.set_ylim(0, 100)
ax_bar.tick_params(axis="x", rotation=10)
ax_bar.grid(axis="y", linestyle="--", alpha=0.35)
ax_bar.set_axisbelow(True)
sns.despine()
plt.tight_layout()
st.pyplot(fig_bar)

st.header("📋 Classification Reports")
selected_model = st.selectbox("Select Model", ["Linear SVM", "Logistic Regression", "Naive Bayes"])

if selected_model == "Linear SVM":
    selected_pred = svm_pred
elif selected_model == "Logistic Regression":
    selected_pred = logistic_pred
else:
    selected_pred = nb_pred

report = classification_report(y_test, selected_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.round(3), use_container_width=True)

st.header("🔲 Confusion Matrix")
classes = sorted(y.unique())
cm = confusion_matrix(y_test, selected_pred, labels=classes)

fig2, ax2 = plt.subplots(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax2)
ax2.set_xlabel("Predicted Category")
ax2.set_ylabel("Actual Category")
ax2.set_title(f"Confusion Matrix - {selected_model}")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
st.pyplot(fig2)

st.header("📄 Predict Legal Document Category")
document_text = st.text_area(
    "Enter legal document text:",
    height=250,
    placeholder="Enter or paste your legal document text here...",
)

if st.button("🔍 Classify Document"):
    if document_text.strip() == "":
        st.warning("Please enter legal document text.")
    else:
        document_vector = tfidf.transform([document_text])
        svm_result = svm_model.predict(document_vector)[0]
        logistic_result = logistic_model.predict(document_vector)[0]
        nb_result = nb_model.predict(document_vector)[0]

        st.success("Document classification completed!")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Linear SVM")
            st.info(svm_result)
        with col2:
            st.subheader("Logistic Regression")
            st.info(logistic_result)
        with col3:
            st.subheader("Naive Bayes")
            st.info(nb_result)

        st.subheader("Classification Summary")
        prediction_table = pd.DataFrame(
            {
                "Algorithm": ["Linear SVM", "Logistic Regression", "Naive Bayes"],
                "Predicted Category": [svm_result, logistic_result, nb_result],
            }
        )
        st.dataframe(prediction_table, use_container_width=True)

st.markdown("---")
st.caption("Legal Document Classification | TF-IDF + Machine Learning")