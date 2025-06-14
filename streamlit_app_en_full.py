
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Prediction Tool for Node-Negative Breast Cancer (pN0)")

@st.cache_data
def load_data():
    df = pd.read_csv("Train_data_english.csv")
    return df

df = load_data()

# Define input features and target
features = ["Age", "Height", "Weight", "Axillary Diagnosis", "Menopause", "Clinical T stage",
            "CNB Histology", "Clinical Histologic Grade", "cER(%)", "cPgR(%)", "cHER2",
            "HER2 Protein", "Tumor size (US)"]
target = ""metastasis"

X = df[features]
y = df[target]

numeric_features = ["Age", "Height", "Weight", "cER(%)", "cPgR(%)", "Tumor size (US)"]
categorical_features = list(set(features) - set(numeric_features))

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=5000))
])
model.fit(X, y)

# User Input
st.header("Patient Information Input")
input_data = {}
for col in features:
    if col in numeric_features:
        input_data[col] = st.number_input(col, value=float(df[col].mean()))
    else:
        input_data[col] = st.selectbox(col, sorted(df[col].dropna().unique()))

input_df = pd.DataFrame([input_data])
pred = model.predict_proba(input_df)[0][1]
st.subheader(f"Predicted Probability of Node-Negative Disease (pN0): **{(1 - pred):.2%}**")

# Feature Importance
st.header("Feature Importance (Logistic Regression Coefficients)")

coefficients = model.named_steps["classifier"].coef_[0]
feature_names = (
    numeric_features +
    list(model.named_steps["preprocessor"]
         .named_transformers_["cat"]
         .named_steps["onehot"]
         .get_feature_names_out(categorical_features))
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", key=abs, ascending=False)

st.dataframe(importance_df)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Coefficient", y="Feature", data=importance_df.head(15), ax=ax)
st.pyplot(fig)
