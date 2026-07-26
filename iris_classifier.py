"""
Project 2 - Data Classification Using AI
DecodeLabs Industrial Training Kit - Batch 2026

Goal: Build a basic classification model using a small dataset (Iris).

Key Requirements covered:
1. LOAD & UNDERSTAND DATASET -> Iris dataset (150 samples, 3 classes, 4 features)
2. TRAIN/TEST SPLIT           -> 80/20 split with shuffling
3. FEATURE SCALING            -> StandardScaler (mean=0, variance=1)
4. CLASSIFICATION ALGORITHM   -> K-Nearest Neighbors (KNN)
5. OUTPUT VALIDATION          -> Confusion Matrix + F1 Score + Accuracy
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import pandas as pd


def load_and_explore_data():
    """
    PHASE 1: INPUT
    Loads the Iris dataset and shows a quick look at it.
    """
    iris = load_iris()
    X = iris.data          # Features: sepal length, sepal width, petal length, petal width
    y = iris.target        # Labels: 0 = setosa, 1 = versicolor, 2 = virginica

    print("=" * 55)
    print(" DATASET OVERVIEW: IRIS ")
    print("=" * 55)
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]} -> {iris.feature_names}")
    print(f"Classes: {list(iris.target_names)}\n")

    # Show first 5 rows as a readable table
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['species'] = [iris.target_names[i] for i in y]
    print("Sample rows:")
    print(df.head(), "\n")

    return X, y, iris.target_names


def split_and_scale(X, y):
    """
    PHASE 2: PROCESS (part 1)
    Splits data into training/testing sets, then scales features.
    """
    # Train-Test Split (80% train, 20% test), shuffled to remove order bias
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Feature Scaling: StandardScaler -> mean=0, variance=1
    # Fit ONLY on training data, then apply same transform to test data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples:  {X_test.shape[0]}\n")

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_and_predict(X_train, X_test, y_train, k=5):
    """
    PHASE 2: PROCESS (part 2)
    Instantiate -> Fit -> Predict, using K-Nearest Neighbors.
    """
    model = KNeighborsClassifier(n_neighbors=k)  # INSTANTIATE
    model.fit(X_train, y_train)                  # FIT (memorize the map)
    predictions = model.predict(X_test)          # PREDICT (apply logic)
    return model, predictions


def evaluate_model(y_test, predictions, target_names):
    """
    PHASE 3: OUTPUT
    Validates model performance with accuracy, confusion matrix, and F1 score.
    """
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')

    print("=" * 55)
    print(" MODEL EVALUATION ")
    print("=" * 55)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"F1 Score (weighted): {f1:.4f}\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions), "\n")

    print("Classification Report:")
    print(classification_report(y_test, predictions, target_names=target_names))


def run_project():
    X, y, target_names = load_and_explore_data()
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    model, predictions = train_and_predict(X_train, X_test, y_train, k=5)
    evaluate_model(y_test, predictions, target_names)


if __name__ == "__main__":
    run_project()
