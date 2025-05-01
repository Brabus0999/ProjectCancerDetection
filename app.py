
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Cancer App", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("global_cancer_patients_2015_2024.csv")

def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Train")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Test")
    plt.title(title)
    plt.xlabel("Training size")
    plt.ylabel("R² Score")
    plt.legend()
    st.pyplot(plt.gcf())

df = load_data()
st.title("Cancer ML Web Application")

model_type = st.sidebar.selectbox("Choose model type", ["Regression", "Clustering", "Classification"])

if model_type == "Regression":
    features = ['Age', 'Gender', 'Country_Region', 'Year', 'Genetic_Risk', 'Air_Pollution',
                'Alcohol_Use', 'Smoking', 'Obesity_Level', 'Cancer_Type', 'Cancer_Stage']
    target = 'Survival_Years'
    X = df[features]
    y = df[target]

    num_cols = ['Age', 'Year', 'Genetic_Risk', 'Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level']
    cat_cols = ['Gender', 'Country_Region', 'Cancer_Type', 'Cancer_Stage']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    model_name = st.selectbox("Choose regression model", ["LinearRegression", "Ridge", "Lasso"])
    if model_name == "LinearRegression":
        model = LinearRegression()
    elif model_name == "Ridge":
        alpha = st.slider("Alpha", 0.01, 10.0, 1.0)
        model = Ridge(alpha=alpha)
    else:
        alpha = st.slider("Alpha", 0.01, 10.0, 0.1)
        model = Lasso(alpha=alpha)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    st.write("MSE:", mean_squared_error(y_test, y_pred))
    st.write("R² Score:", r2_score(y_test, y_pred))
    plot_learning_curve(pipeline, X, y, f"Learning Curve - {model_name}")

elif model_type == "Clustering":
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    scaled = StandardScaler().fit_transform(numeric_df.drop(columns=["Year"]))
    k = st.slider("Number of Clusters (k)", 2, 10, 5)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = model.fit_predict(scaled)
    st.write("Silhouette Score:", silhouette_score(scaled, df['Cluster']))

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled)
    df['PCA1'], df['PCA2'] = pca_data[:, 0], pca_data[:, 1]

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', ax=ax)
    ax.set_title("PCA Clusters")
    st.pyplot(fig)

elif model_type == "Classification":
    df['Severity_Class'] = df['Target_Severity_Score'].apply(lambda x: 0 if x < 3.5 else (1 if x < 5 else 2))
    df_model = df.drop(columns=["Patient_ID", "Target_Severity_Score"])
    cat_cols = ['Gender', 'Country_Region', 'Cancer_Type', 'Cancer_Stage']
    for col in cat_cols:
        df_model[col] = LabelEncoder().fit_transform(df_model[col])

    X = df_model.drop(columns=["Severity_Class"])
    y = df_model["Severity_Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())
