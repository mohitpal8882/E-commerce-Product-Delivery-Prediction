import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="E-commerce Delivery Prediction", layout="wide")

st.title("📦 E-commerce Product Delivery Prediction")
st.markdown("Predict whether a product will be delivered **On Time** or **Late**")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Train 2.csv")
    return df

df = load_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["Dataset Overview", "EDA", "Model Training", "Prediction"]
)

# --------------------------------------------------
# DATA PREPROCESSING
# --------------------------------------------------
df_processed = df.copy()
df_processed.drop("ID", axis=1, inplace=True)

le = LabelEncoder()
cat_cols = ["Warehouse_block", "Mode_of_Shipment", "Product_importance", "Gender"]

for col in cat_cols:
    df_processed[col] = le.fit_transform(df_processed[col])

X = df_processed.drop("Reached.on.Time_Y.N", axis=1)
y = df_processed["Reached.on.Time_Y.N"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

lr.fit(X_train_scaled, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
knn.fit(X_train_scaled, y_train)

# --------------------------------------------------
# MENU: DATASET OVERVIEW
# --------------------------------------------------
if menu == "Dataset Overview":
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Target Variable Distribution")
    st.bar_chart(df["Reached.on.Time_Y.N"].value_counts())

# --------------------------------------------------
# MENU: EDA
# --------------------------------------------------
elif menu == "EDA":
    st.subheader("🔍 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.countplot(x="Mode_of_Shipment", hue="Reached.on.Time_Y.N", data=df, ax=ax)
        plt.title("Shipment Mode vs Delivery")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.countplot(x="Product_importance", hue="Reached.on.Time_Y.N", data=df, ax=ax)
        plt.title("Product Importance vs Delivery")
        st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.boxplot(x="Reached.on.Time_Y.N", y="Customer_care_calls", data=df, ax=ax)
    plt.title("Customer Care Calls vs Delivery")
    st.pyplot(fig)

# --------------------------------------------------
# MENU: MODEL TRAINING
# --------------------------------------------------
elif menu == "Model Training":
    st.subheader("⚙ Model Performance")

    results = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "KNN"],
        "Accuracy": [
            accuracy_score(y_test, lr.predict(X_test_scaled)),
            accuracy_score(y_test, dt.predict(X_test)),
            accuracy_score(y_test, rf.predict(X_test)),
            accuracy_score(y_test, knn.predict(X_test_scaled))
        ]
    })

    st.dataframe(results)

    fig, ax = plt.subplots()
    sns.barplot(x="Model", y="Accuracy", data=results, ax=ax)
    plt.xticks(rotation=30)
    plt.title("Model Accuracy Comparison")
    st.pyplot(fig)

    st.subheader("🌲 Feature Importance (Random Forest)")
    fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    fig, ax = plt.subplots()
    fi.plot(kind="bar", ax=ax)
    plt.title("Feature Importance")
    st.pyplot(fig)

# --------------------------------------------------
# MENU: PREDICTION
# --------------------------------------------------
elif menu == "Prediction":
    st.subheader("🧠 Predict Delivery Status")

    warehouse = st.selectbox("Warehouse Block", df["Warehouse_block"].unique())
    shipment = st.selectbox("Mode of Shipment", df["Mode_of_Shipment"].unique())
    gender = st.selectbox("Gender", df["Gender"].unique())
    importance = st.selectbox("Product Importance", df["Product_importance"].unique())

    care_calls = st.number_input("Customer Care Calls", 0, 10, 2)
    rating = st.slider("Customer Rating", 1, 5, 3)
    cost = st.number_input("Cost of Product", 100, 5000, 500)
    discount = st.slider("Discount Offered (%)", 0, 80, 10)
    weight = st.number_input("Weight in grams", 500, 5000, 2000)

    input_df = pd.DataFrame([{
        "Warehouse_block": warehouse,
        "Mode_of_Shipment": shipment,
        "Customer_care_calls": care_calls,
        "Customer_rating": rating,
        "Cost_of_the_Product": cost,
        "Prior_purchases": 3,
        "Product_importance": importance,
        "Gender": gender,
        "Discount_offered": discount,
        "Weight_in_gms": weight
    }])

    for col in cat_cols:
        input_df[col] = le.fit_transform(input_df[col])

    input_scaled = scaler.transform(input_df)

    if st.button("Predict"):
        prediction = rf.predict(input_df)[0]

        if prediction == 1:
            st.error("🚨 Late Delivery Predicted")
        else:
            st.success("✅ On-Time Delivery Predicted")
