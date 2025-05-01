
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# Config & Data
# --------------------------------------------------
st.set_page_config(page_title="Cancer ML App", layout="wide")

@st.cache_data
def load_data(path: str = "global_cancer_patients_2015_2024.csv") -> pd.DataFrame:
    """Load the dataset (cached)."""
    return pd.read_csv(path)

df = load_data()

# --------------------------------------------------
# Helper plots
# --------------------------------------------------
def plot_learning_curve(estimator, X, y, title: str):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    fig, ax = plt.subplots()
    ax.plot(train_sizes, np.mean(train_scores, axis=1), label="Train")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), label="Test")
    ax.set_xlabel("Training size")
    ax.set_ylabel("R² Score")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

def show_classification_report(y_true, y_pred):
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report_dict).transpose())

# --------------------------------------------------
# Regression
# --------------------------------------------------
def regression_app(data: pd.DataFrame):
    st.header("🧮 Survival Years Regression")
    features = [
        'Age', 'Gender', 'Country_Region', 'Year', 'Genetic_Risk', 'Air_Pollution',
        'Alcohol_Use', 'Smoking', 'Obesity_Level', 'Cancer_Type', 'Cancer_Stage'
    ]
    target = 'Survival_Years'

    X = data[features]
    y = data[target]

    num_cols = ['Age', 'Year', 'Genetic_Risk', 'Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level']
    cat_cols = ['Gender', 'Country_Region', 'Cancer_Type', 'Cancer_Stage']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    model_name = st.sidebar.radio("Select regression model", ["LinearRegression", "Ridge", "Lasso"])

    if model_name == "LinearRegression":
        model = LinearRegression()
    else:
        alpha = st.sidebar.slider("Alpha", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        model = Ridge(alpha=alpha) if model_name == "Ridge" else Lasso(alpha=alpha)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    test_size = st.sidebar.slider("Test size (fraction)", 0.1, 0.4, 0.2, 0.05)
    if st.sidebar.button("Train Regression Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Results")
        st.write(f"**MSE:** {mse:.3f}")
        st.write(f"**R² Score:** {r2:.3f}")

        st.subheader("Learning Curve")
        plot_learning_curve(pipeline, X, y, f"Learning Curve - {model_name}")

# --------------------------------------------------
# Clustering
# --------------------------------------------------
def clustering_app(data: pd.DataFrame):
    st.header("🔗 Patient Clustering (K‑Means)")
    numeric_df = data.select_dtypes(include=["float64", "int64"]).drop(columns=["Year"], errors='ignore')

    # Scale
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)

    k = st.sidebar.slider("Number of clusters (k)", 2, 10, 5, 1)

    if st.sidebar.button("Run Clustering"):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = km.fit_predict(scaled)
        data_clust = data.copy()
        data_clust['Cluster'] = clusters

        sil_score = None
        try:
            from sklearn.metrics import silhouette_score
            sil_score = silhouette_score(scaled, clusters)
        except Exception:
            pass

        st.subheader("Silhouette Score")
        if sil_score:
            st.write(f"{sil_score:.3f}")

        # PCA for 2D plot
        pca = PCA(n_components=2, random_state=42)
        pca_data = pca.fit_transform(scaled)
        data_clust['PCA1'] = pca_data[:, 0]
        data_clust['PCA2'] = pca_data[:, 1]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=data_clust, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', ax=ax)
        ax.set_title("PCA Projection of Clusters")
        st.pyplot(fig)

# --------------------------------------------------
# Classification
# --------------------------------------------------
def classification_app(data: pd.DataFrame):
    st.header("🩸 Severity Level Classification (Random Forest)")
    df_cls = data.copy()
    df_cls['Severity_Class'] = df_cls['Target_Severity_Score'].apply(
        lambda x: 0 if x < 3.5 else (1 if x < 5 else 2)
    )

    df_model = df_cls.drop(columns=["Patient_ID", "Target_Severity_Score"], errors='ignore')

    cat_cols = ['Gender', 'Country_Region', 'Cancer_Type', 'Cancer_Stage']
    for col in cat_cols:
        df_model[col] = LabelEncoder().fit_transform(df_model[col])

    X = df_model.drop(columns=["Severity_Class"])
    y = df_model["Severity_Class"]

    test_size = st.sidebar.slider("Test size (fraction)", 0.1, 0.4, 0.2, 0.05)
    n_estimators = st.sidebar.slider("Number of trees", 50, 500, 100, 25)

    if st.sidebar.button("Train Classifier"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.subheader("Accuracy")
        st.write(f"{acc:.3f}")

        st.subheader("Classification Report")
        show_classification_report(y_test, y_pred)

# --------------------------------------------------
# Main app dispatcher
# --------------------------------------------------
def main():
    st.sidebar.title("ML Task")
    task = st.sidebar.selectbox("Select task", ["Regression", "Clustering", "Classification"])

    if task == "Regression":
        regression_app(df)
    elif task == "Clustering":
        clustering_app(df)
    else:
        classification_app(df)

if __name__ == "__main__":
    main()
