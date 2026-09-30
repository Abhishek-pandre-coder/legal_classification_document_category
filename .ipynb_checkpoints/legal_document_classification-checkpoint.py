import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Legal Document Classification",
    page_icon="⚖️",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #1f3c88;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 30px;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.stButton>button {
    width: 100%;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">⚖️ Legal Document Category Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'TF-IDF + Linear SVM, Logistic Regression, and Naive Bayes'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Project Settings")

st.sidebar.markdown("""
### Machine Learning Pipeline

1. Upload Dataset
2. Select Text Column
3. Select Category Column
4. Preprocess Text
5. Apply TF-IDF
6. Train Models
7. Evaluate Models
8. Predict Documents
""")

# ============================================================
# DATASET UPLOAD
# ============================================================

st.header("📂 Step 1: Upload Legal Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

default_dataset = Path(__file__).resolve().parent / "legal_documents.csv"

if uploaded_file is None and not default_dataset.exists():

    st.info(
        "Please upload your legal document dataset "
        "to start classification."
    )

    st.markdown("""
    ### Expected Dataset Format

    Your dataset should contain at least two columns:

    | document_text | category |
    |---|---|
    | Agreement between two parties... | Contract |
    | The court hereby orders... | Court |
    | This statute provides... | Statute |

    **Text column:** contains the legal document text.

    **Category column:** contains the document category/class.
    """)

    st.stop()

# ============================================================
# READ DATASET
# ============================================================

try:

    dataset_source = uploaded_file or default_dataset
    dataset_name = (
        uploaded_file.name
        if uploaded_file is not None
        else default_dataset.name
    )

    if Path(dataset_name).suffix.lower() == ".csv":
        df = pd.read_csv(dataset_source)

    else:
        df = pd.read_excel(dataset_source)

    if uploaded_file is None:
        st.success(f"Default dataset loaded: {default_dataset.name}")
    else:
        st.success("Uploaded dataset loaded successfully.")

except Exception as e:

    st.error(f"Error reading dataset: {e}")
    st.stop()

# ============================================================
# DATASET INFORMATION
# ============================================================

st.header("📊 Step 2: Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Documents", len(df))

with col2:
    st.metric("Total Columns", len(df.columns))

with col3:
    st.metric("Missing Values", int(df.isnull().sum().sum()))

with col4:
    st.metric("Duplicate Rows", int(df.duplicated().sum()))

st.subheader("Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)

# ============================================================
# COLUMN SELECTION
# ============================================================

st.header("📝 Step 3: Select Dataset Columns")

columns = df.columns.tolist()

text_default_index = next(
    (
        index for index, column in enumerate(columns)
        if str(column).strip().lower() in {"text", "document", "document_text"}
    ),
    0
)
category_default_index = next(
    (
        index for index, column in enumerate(columns)
        if str(column).strip().lower() in {"category", "label", "class"}
        and index != text_default_index
    ),
    1 if len(columns) > 1 and text_default_index != 1 else 0
)

col1, col2 = st.columns(2)

with col1:

    text_column = st.selectbox(
        "Select Legal Document Text Column",
        columns,
        index=text_default_index
    )

with col2:

    category_column = st.selectbox(
        "Select Category / Target Column",
        columns,
        index=category_default_index
    )

if text_column == category_column:
    st.error("Select different columns for document text and category.")
    st.stop()

# ============================================================
# PREPROCESSING
# ============================================================

st.header("🧹 Step 4: Data Preprocessing")

data = df[[text_column, category_column]].copy()

data.columns = ["text", "category"]

# Remove missing values

data.dropna(inplace=True)

# Convert text to string

data["text"] = data["text"].astype(str)

# Remove empty text

data = data[data["text"].str.strip() != ""]

# Remove duplicate documents

data.drop_duplicates(inplace=True)

# Convert category to string

data["category"] = data["category"].astype(str)

st.success(
    f"After preprocessing: {len(data)} documents available."
)

st.subheader("📄 Select Document Text and Content")

selected_document_index = st.selectbox(
    "Choose a document to view",
    data.index,
    format_func=lambda index: (
        f"Document {index + 1} - {data.loc[index, 'category']}"
    )
)

st.info(f"Category: {data.loc[selected_document_index, 'category']}")
st.text_area(
    "Selected Legal Document Text",
    value=data.loc[selected_document_index, "text"],
    height=160,
    key="selected_document_content"
)

# ============================================================
# CATEGORY DISTRIBUTION
# ============================================================

st.subheader("📚 Legal Document Category Distribution")

category_counts = data["category"].value_counts()

st.dataframe(
    category_counts.rename("Document Count"),
    use_container_width=True
)

fig, ax = plt.subplots(figsize=(10, 5))

sns.barplot(
    x=category_counts.index,
    y=category_counts.values,
    ax=ax
)

ax.set_xlabel("Legal Document Category")
ax.set_ylabel("Number of Documents")
ax.set_title("Category Distribution")

plt.xticks(rotation=45)

st.pyplot(fig)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

st.header("🔀 Step 5: Train-Test Split")

test_size = st.slider(
    "Test Size",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05
)

random_state = st.number_input(
    "Random State",
    min_value=1,
    max_value=100,
    value=42
)

X = data["text"]
y = data["category"]

# Check class count

if y.nunique() < 2:

    st.error(
        "Classification requires at least two categories."
    )

    st.stop()

try:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

except ValueError:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

st.write(
    f"Training documents: **{len(X_train)}**"
)

st.write(
    f"Testing documents: **{len(X_test)}**"
)

# ============================================================
# TF-IDF
# ============================================================

st.header("🔢 Step 6: TF-IDF Feature Extraction")

max_features = st.slider(
    "Maximum TF-IDF Features",
    min_value=1000,
    max_value=30000,
    value=10000,
    step=1000
)

ngram_choice = st.selectbox(
    "Select N-gram Range",
    [
        "Unigram (1,1)",
        "Unigram + Bigram (1,2)"
    ]
)

if ngram_choice == "Unigram (1,1)":
    ngram_range = (1, 1)
else:
    ngram_range = (1, 2)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=max_features,
    ngram_range=ngram_range,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

st.success(
    f"TF-IDF completed. Feature matrix: "
    f"{X_train_tfidf.shape[0]} × {X_train_tfidf.shape[1]}"
)

# ============================================================
# MODEL TRAINING
# ============================================================

st.header("🤖 Step 7: Train Machine Learning Models")

train_button = st.button(
    "🚀 Train All Models",
    type="primary"
)

if train_button:

    # --------------------------------------------------------
    # Linear SVM
    # --------------------------------------------------------

    svm_model = LinearSVC(
        random_state=random_state
    )

    svm_model.fit(
        X_train_tfidf,
        y_train
    )

    svm_pred = svm_model.predict(
        X_test_tfidf
    )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    lr_model = LogisticRegression(
        max_iter=2000,
        random_state=random_state
    )

    lr_model.fit(
        X_train_tfidf,
        y_train
    )

    lr_pred = lr_model.predict(
        X_test_tfidf
    )

    # --------------------------------------------------------
    # Naive Bayes
    # --------------------------------------------------------

    nb_model = MultinomialNB()

    nb_model.fit(
        X_train_tfidf,
        y_train
    )

    nb_pred = nb_model.predict(
        X_test_tfidf
    )

    # --------------------------------------------------------
    # SAVE MODELS IN SESSION
    # --------------------------------------------------------

    st.session_state["vectorizer"] = vectorizer

    st.session_state["svm_model"] = svm_model
    st.session_state["lr_model"] = lr_model
    st.session_state["nb_model"] = nb_model

    st.session_state["svm_pred"] = svm_pred
    st.session_state["lr_pred"] = lr_pred
    st.session_state["nb_pred"] = nb_pred

    st.session_state["y_test"] = y_test

    st.success(
        "All three machine learning models trained successfully!"
    )

# ============================================================
# EVALUATION
# ============================================================

if "svm_model" in st.session_state:

    st.header("📈 Step 8: Model Performance Comparison")

    y_test = st.session_state["y_test"]

    svm_pred = st.session_state["svm_pred"]
    lr_pred = st.session_state["lr_pred"]
    nb_pred = st.session_state["nb_pred"]

    # Metrics

    models = [
        "Linear SVM",
        "Logistic Regression",
        "Naive Bayes"
    ]

    predictions = [
        svm_pred,
        lr_pred,
        nb_pred
    ]

    results = []

    for name, pred in zip(models, predictions):

        results.append({

            "Model": name,

            "Accuracy": accuracy_score(
                y_test,
                pred
            ),

            "Precision": precision_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0
            ),

            "Recall": recall_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0
            ),

            "F1 Score": f1_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0
            )
        })

    results_df = pd.DataFrame(results)

    # Percentage

    metric_df = results_df.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]:

        metric_df[column] = (
            metric_df[column] * 100
        ).round(2)

    st.dataframe(
        metric_df,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Accuracy Chart
    # --------------------------------------------------------

    st.subheader("📊 Accuracy Comparison")

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.barplot(
        data=metric_df,
        x="Model",
        y="Accuracy",
        ax=ax
    )

    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)

    for container in ax.containers:

        ax.bar_label(
            container,
            fmt="%.2f%%"
        )

    st.pyplot(fig)

    # --------------------------------------------------------
    # COMPLETE METRIC CHART
    # --------------------------------------------------------

    st.subheader("📊 Model Performance")

    plot_df = metric_df.set_index("Model")

    fig, ax = plt.subplots(figsize=(11, 6))

    plot_df[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    ].plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)

    plt.xticks(rotation=0)

    st.pyplot(fig)

    # ========================================================
    # DETAILED CLASSIFICATION REPORT
    # ========================================================

    st.header("📋 Step 9: Classification Reports")

    selected_model = st.selectbox(
        "Select Model",
        models
    )

    if selected_model == "Linear SVM":

        selected_pred = svm_pred

    elif selected_model == "Logistic Regression":

        selected_pred = lr_pred

    else:

        selected_pred = nb_pred

    report = classification_report(
        y_test,
        selected_pred,
        zero_division=0
    )

    st.code(report)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.header("🔲 Step 10: Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        selected_pred
    )

    labels = sorted(
        list(set(y_test))
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_title(
        f"{selected_model} - Confusion Matrix"
    )

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    st.pyplot(fig)

    # ========================================================
    # DOCUMENT PREDICTION
    # ========================================================

    st.header("📄 Step 11: Predict Legal Document Category")

    input_text = st.text_area(
        "Enter legal document text:",
        height=250,
        placeholder=(
            "Enter a legal document, court order, "
            "contract, statute, judgment, etc."
        )
    )

    prediction_model = st.selectbox(
        "Choose Prediction Model",
        models,
        key="prediction_model"
    )

    if st.button(
        "🔍 Predict Category",
        type="primary"
    ):

        if input_text.strip() == "":

            st.warning(
                "Please enter some legal document text."
            )

        else:

            text_vector = st.session_state[
                "vectorizer"
            ].transform(
                [input_text]
            )

            if prediction_model == "Linear SVM":

                prediction = st.session_state[
                    "svm_model"
                ].predict(text_vector)[0]

            elif prediction_model == "Logistic Regression":

                prediction = st.session_state[
                    "lr_model"
                ].predict(text_vector)[0]

            else:

                prediction = st.session_state[
                    "nb_model"
                ].predict(text_vector)[0]

            st.success(
                f"Predicted Legal Document Category: **{prediction}**"
            )

    # ========================================================
    # BATCH PREDICTION
    # ========================================================

    st.header("📁 Step 12: Batch Prediction")

    batch_file = st.file_uploader(
        "Upload CSV containing a text column",
        type=["csv"],
        key="batch_upload"
    )

    if batch_file is not None:

        batch_df = pd.read_csv(batch_file)

        batch_text_column = st.selectbox(
            "Select text column for batch prediction",
            batch_df.columns
        )

        if st.button(
            "🚀 Predict Uploaded Documents"
        ):

            batch_vectors = st.session_state[
                "vectorizer"
            ].transform(
                batch_df[
                    batch_text_column
                ].astype(str)
            )

            batch_df["Predicted_Category"] = (
                st.session_state["svm_model"]
                .predict(batch_vectors)
            )

            st.subheader(
                "Prediction Results"
            )

            st.dataframe(
                batch_df,
                use_container_width=True
            )

            csv_data = batch_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Predictions",
                data=csv_data,
                file_name="legal_document_predictions.csv",
                mime="text/csv"
            )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <center>
    <b>Legal Document Category Classification</b><br>
    TF-IDF + Linear SVM + Logistic Regression + Naive Bayes<br>
    Machine Learning / NLP Project
    </center>
    """,
    unsafe_allow_html=True
)