# model_pipeline.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

def preprocess_and_train():
    # Load dataset
    df = pd.read_csv("adult 3.csv")
    
    # Handle missing values
    df.replace(" ?", np.nan, inplace=True)
    df.dropna(inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        if column != 'income':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

    # Encode target variable
    df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

    # Split dataset
    X = df.drop("income", axis=1)
    y = df["income"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train KNN model
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # Save model and preprocessing tools
    joblib.dump(model, "knn_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoders, "label_encoders.pkl")

if __name__ == "__main__":
    preprocess_and_train()
    print("✅ Model and preprocessing saved successfully.")
