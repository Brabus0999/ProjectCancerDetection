#!/usr/bin/env python
# coding: utf-8

# In[242]:


import pandas as pd
import numpy as np


# In[243]:


df = pd.read_csv('global_cancer_patients_2015_2024.csv')
df


# In[244]:


df.info()


# In[245]:


df.head()


# In[246]:


df.describe()


# In[247]:


df.isnull().sum()


# In[248]:


df_cleaned = df.dropna()


# In[249]:


df_cleaned.duplicated().sum()


# In[250]:


Q1 = df_cleaned['Age'].quantile(0.25)
Q3 = df_cleaned['Age'].quantile(0.75)
IQR = Q3 - Q1


# In[251]:


df_cleaned = df_cleaned[(df_cleaned['Age'] >= Q1 - 1.5 * IQR) & (df_cleaned['Age'] <= Q3 + 1.5 * IQR)]


# In[252]:


df_cleaned.select_dtypes(include='object').columns


# In[253]:


df_encoded = pd.get_dummies(df_cleaned, drop_first=True)


# In[254]:


from sklearn.preprocessing import StandardScaler


# In[255]:


numerical_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns


# In[256]:


scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])


# In[257]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[258]:


features = [
    'Age', 'Gender', 'Country_Region', 'Year', 'Genetic_Risk', 'Air_Pollution',
    'Alcohol_Use', 'Smoking', 'Obesity_Level', 'Cancer_Type', 'Cancer_Stage'
]
target = 'Survival_Years'

X = df[features]
y = df[target]


# In[259]:


numerical_features = ['Age', 'Year', 'Genetic_Risk', 'Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level']
categorical_features = ['Gender', 'Country_Region', 'Cancer_Type', 'Cancer_Stage']


# In[260]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)


# In[261]:


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])


# In[262]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[263]:


pipeline.fit(X_train, y_train)


# In[264]:


y_pred = pipeline.predict(X_test)


# In[265]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse, r2


# In[266]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# In[267]:


df = pd.read_csv("global_cancer_patients_2015_2024.csv")


# In[268]:


features = [
    'Age', 'Gender', 'Country_Region', 'Year', 'Genetic_Risk', 'Air_Pollution',
    'Alcohol_Use', 'Smoking', 'Obesity_Level', 'Cancer_Type', 'Cancer_Stage'
]
target = 'Survival_Years'
X = df[features]
y = df[target]


# In[269]:


numerical = ['Age', 'Year', 'Genetic_Risk', 'Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level']
categorical = ['Gender', 'Country_Region', 'Cancer_Type', 'Cancer_Stage']
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
])


# In[270]:


ridge = Pipeline([('preprocessor', preprocessor), ('regressor', Ridge(alpha=1.0))])
lasso = Pipeline([('preprocessor', preprocessor), ('regressor', Lasso(alpha=0.1))])


# In[271]:


def plot_learning_curve(model, title):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=3, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1)
    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Train R²')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Test R²')
    plt.title(title)
    plt.xlabel('Training Set Size')
    plt.ylabel('R² Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[272]:


plot_learning_curve(ridge, "Ridge Regression")
plot_learning_curve(lasso, "Lasso Regression")


# In[273]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error


# In[274]:


df_model = df.drop(columns=["Patient_ID", "Cancer_Type", "Cancer_Stage", "Country_Region", "Gender"])
df_model = df_model.dropna()
X = df_model.drop(columns=["Target_Severity_Score"])
y = df_model["Target_Severity_Score"]


# In[275]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[276]:


ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])
ridge_params = {'ridge__alpha': [0.01, 0.1, 1, 10, 100]}
ridge_grid = GridSearchCV(ridge_pipeline, ridge_params, cv=5, scoring='r2')
ridge_grid.fit(X_train, y_train)


# In[277]:


ridge_best = ridge_grid.best_estimator_
y_pred = ridge_best.predict(X_test)
print("Best Alpha:", ridge_grid.best_params_['ridge__alpha'])
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))


# In[278]:


from sklearn.linear_model import Lasso


# In[279]:


lasso_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(max_iter=10000))
])
lasso_params = {'lasso__alpha': [0.01, 0.1, 1, 10, 100]}
lasso_grid = GridSearchCV(lasso_pipeline, lasso_params, cv=5, scoring='r2')
lasso_grid.fit(X_train, y_train)


# In[280]:


lasso_best = lasso_grid.best_estimator_
lasso_pred = lasso_best.predict(X_test)

print("Best Alpha (Lasso):", lasso_grid.best_params_['lasso__alpha'])
print("R2 Score (Lasso):", r2_score(y_test, lasso_pred))
print("MSE (Lasso):", mean_squared_error(y_test, lasso_pred))


# In[281]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# In[282]:


df = pd.read_csv("global_cancer_patients_2015_2024.csv")


# In[283]:


numeric_df = df.select_dtypes(include=["int64", "float64"])


# In[284]:


scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)


# In[285]:


kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_data)


# In[286]:


pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)
df['PCA1'] = pca_components[:, 0]
df['PCA2'] = pca_components[:, 1]


# In[287]:


plt.figure(figsize=(8, 6))
for cluster in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}', alpha=0.6)
plt.title('K-Means Clusters (k=5) Visualized with PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[288]:


cluster_summary = df.groupby('Cluster')[numeric_df.columns].mean().round(2)
display(cluster_summary)


# In[289]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[290]:


df_model = df.drop(columns=['Patient_ID', 'Target_Severity_Score'])


# In[291]:


categorical_cols = ['Gender', 'Country_Region', 'Cancer_Type', 'Cancer_Stage']
df_model[categorical_cols] = df_model[categorical_cols].apply(LabelEncoder().fit_transform)


# In[292]:


def categorize_severity(score):
    if score < 3.5:
        return 0  
    elif score < 5.0:
        return 1  
    else:
        return 2 

df['Severity_Class'] = df['Target_Severity_Score'].apply(categorize_severity)


# In[293]:


from sklearn.preprocessing import LabelEncoder

df_model = df.drop(columns=['Patient_ID', 'Target_Severity_Score'])

categorical_cols = ['Gender', 'Country_Region', 'Cancer_Type', 'Cancer_Stage']
df_model[categorical_cols] = df_model[categorical_cols].apply(LabelEncoder().fit_transform)

X = df_model.drop(columns=['Severity_Class'])
y = df_model['Severity_Class']


# In[294]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[295]:


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# In[296]:


y_pred = rf_model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)


# In[297]:


import pandas as pd
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

report_df


# In[298]:


from sklearn.metrics import accuracy_score


# In[299]:


y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)


# In[300]:


train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)


# In[301]:


print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy:     {test_accuracy:.4f}")


# In[302]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


# In[303]:


train_sizes, train_scores, test_scores = learning_curve(
    rf_model, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
)


# In[304]:


train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)


# In[305]:


plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, label='Training Accuracy')
plt.plot(train_sizes, test_scores_mean, label='Validation Accuracy')
plt.title('Learning Curve for Random Forest')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[306]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# In[307]:


param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}


# In[308]:


rf_base = RandomForestClassifier(random_state=42)


# In[309]:


grid_search = GridSearchCV(estimator=rf_base,
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=1)


# In[310]:


grid_search.fit(X_train, y_train)


# In[311]:


best_rf = grid_search.best_estimator_


# In[312]:


print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)


# In[313]:


from sklearn.metrics import classification_report


# In[314]:


y_best_pred = best_rf.predict(X_test)


# In[315]:


print(classification_report(y_test, y_best_pred))


# In[316]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# In[317]:


y_pred = best_rf.predict(X_test)


# In[318]:


cm = confusion_matrix(y_test, y_pred)


# In[319]:


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_rf.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix for Random Forest')
plt.grid(False)
plt.tight_layout()
plt.show()


# In[320]:


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


# In[321]:


st.set_page_config(page_title="ML Cancer App", layout="wide")


# In[322]:


@st.cache_data
def load_data():
    return pd.read_csv("global_cancer_patients_2015_2024.csv")


# In[323]:


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


# In[324]:


df = load_data()
st.title("Cancer ML Web Application")


# In[325]:


model_type = st.sidebar.selectbox("Choose model type", ["Regression", "Clustering", "Classification"])


# In[326]:


if model_type == "Regression":
  
    features = ['Age', 'Gender', 'Country_Region', 'Year', 'Genetic_Risk', 'Air_Pollution',
                'Alcohol_Use', 'Smoking', 'Obesity_Level', 'Cancer_Type', 'Cancer_Stage']
    target = 'Survival_Years'
    X = df[features]
    y = df[target]


# In[327]:


num_cols = ['Age', 'Year', 'Genetic_Risk', 'Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level']
cat_cols = ['Gender', 'Country_Region', 'Cancer_Type', 'Cancer_Stage']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])


# In[328]:


model_name = "Ridge"


# In[329]:


alpha = 1.0 


# In[330]:


if model_name == "LinearRegression":
    model = LinearRegression()
elif model_name == "Ridge":
    model = Ridge(alpha=alpha)
else:
    model = Lasso(alpha=alpha)


# In[331]:


pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])


# In[332]:


from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np


# In[333]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)


# In[334]:


y_pred = pipeline.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


# In[335]:


train_sizes, train_scores, test_scores = learning_curve(
    pipeline, X, y, cv=5, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
)

plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Train")
plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Test")
plt.title(f"Learning Curve - {model_name}")
plt.xlabel("Training Size")
plt.ylabel("R² Score")
plt.legend()
plt.grid(True)
plt.show()


# In[336]:


from sklearn.preprocessing import StandardScaler

numeric_df = df.select_dtypes(include=["float64", "int64"])
scaled = StandardScaler().fit_transform(numeric_df.drop(columns=["Year"]))


# In[337]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

k = 5  

model = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = model.fit_predict(scaled)

score = silhouette_score(scaled, df['Cluster'])
print("Silhouette Score:", score)


# In[338]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled)
df['PCA1'] = pca_data[:, 0]
df['PCA2'] = pca_data[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10')
plt.title("PCA Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()


# In[339]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd


# In[340]:


df['Severity_Class'] = df['Target_Severity_Score'].apply(lambda x: 0 if x < 3.5 else (1 if x < 5 else 2))


# In[341]:


df_model = df.drop(columns=["Patient_ID", "Target_Severity_Score"])


# In[342]:


cat_cols = ['Gender', 'Country_Region', 'Cancer_Type', 'Cancer_Stage']
for col in cat_cols:
    df_model[col] = LabelEncoder().fit_transform(df_model[col])


# In[343]:


X = df_model.drop(columns=["Severity_Class"])
y = df_model["Severity_Class"]


# In[344]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[345]:


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# In[346]:


print("Accuracy:", accuracy_score(y_test, y_pred))
report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
display(report_df)


# In[ ]:




