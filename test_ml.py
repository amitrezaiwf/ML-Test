# app.py

import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

def train_model():
    # Features and target
    X = iris_df.drop('target', axis=1)
    y = iris_df['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

# Train the model
trained_model, accuracy = train_model()

# Streamlit web app
st.title("Iris Flower Classification App")

# Add input features (features of the Iris flower)
sepal_length = st.slider("Sepal Length", float(iris_df['sepal length (cm)'].min()), float(iris_df['sepal length (cm)'].max()), float(iris_df['sepal length (cm)'].mean()))
sepal_width = st.slider("Sepal Width", float(iris_df['sepal width (cm)'].min()), float(iris_df['sepal width (cm)'].max()), float(iris_df['sepal width (cm)'].mean()))
petal_length = st.slider("Petal Length", float(iris_df['petal length (cm)'].min()), float(iris_df['petal length (cm)'].max()), float(iris_df['petal length (cm)'].mean()))
petal_width = st.slider("Petal Width", float(iris_df['petal width (cm)'].min()), float(iris_df['petal width (cm)'].max()), float(iris_df['petal width (cm)'].mean()))

# Make predictions
prediction = trained_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

# Display the prediction
st.write(f"Predicted Iris Species: {iris.target_names[prediction[0]]}")

# Display model accuracy
st.write(f"Model Accuracy: {accuracy:.2%}")

