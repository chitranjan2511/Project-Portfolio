"""
Telco Customer Churn Analysis
==============================
Big Data Analysis using PySpark, Scikit-learn, PyTorch, and Keras.
Includes EDA, preprocessing, classical ML models, and deep learning models.
"""

# ─────────────────────────────────────────────
# 1. Imports
# ─────────────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

import tensorflow as tf
from tensorflow import keras

from pyspark.sql import SparkSession

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ─────────────────────────────────────────────
# 2. Configuration
# ─────────────────────────────────────────────
FILE_PATH = "Telco_Customer_Churn.csv"   # ← update path if needed
RANDOM_STATE = 42
TEST_SIZE = 0.2
PYTORCH_EPOCHS = 100
PYTORCH_LR = 0.01
KERAS_EPOCHS = 20
KERAS_BATCH_SIZE = 32

# ─────────────────────────────────────────────
# 3. Spark Session & Data Loading
# ─────────────────────────────────────────────
def load_data(file_path: str) -> pd.DataFrame:
    """Load the CSV via PySpark and return a Pandas DataFrame."""
    spark = (
        SparkSession.builder
        .appName("Telco Churn Analysis")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    df_spark = spark.read.csv(file_path, header=True, inferSchema=True)
    df = df_spark.toPandas()
    spark.stop()
    return df


# ─────────────────────────────────────────────
# 4. Exploratory Data Analysis (EDA)
# ─────────────────────────────────────────────
def run_eda(df: pd.DataFrame) -> None:
    """Print summary statistics and render key visualisation plots."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape : {df.shape}")
    print(f"\nColumn dtypes:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nChurn distribution:\n{df['Churn'].value_counts()}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Exploratory Data Analysis", fontsize=16, fontweight="bold")

    # Churn distribution
    df["Churn"].value_counts().plot(kind="bar", ax=axes[0], color=["steelblue", "tomato"])
    axes[0].set_title("Churn Distribution")
    axes[0].set_xlabel("Churn")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)

    # Monthly charges by churn
    df.boxplot(column="MonthlyCharges", by="Churn", ax=axes[1])
    axes[1].set_title("Monthly Charges by Churn")
    axes[1].set_xlabel("Churn")
    axes[1].set_ylabel("Monthly Charges")

    # Tenure distribution
    df["tenure"].hist(bins=30, ax=axes[2], color="steelblue", edgecolor="white")
    axes[2].set_title("Tenure Distribution")
    axes[2].set_xlabel("Tenure (months)")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("EDA plots saved → eda_plots.png")


# ─────────────────────────────────────────────
# 5. Preprocessing
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """
    Clean and encode the dataset.
    Returns: X_train, X_test, y_train, y_test,
             X_train_scaled, X_test_scaled, feature_names
    """
    data = df.copy()

    # Drop non-informative ID column
    data.drop(columns=["customerID"], inplace=True, errors="ignore")

    # Fix TotalCharges (can be whitespace string)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"].fillna(data["TotalCharges"].median(), inplace=True)

    # Encode binary target
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

    # Label-encode all remaining object columns
    le = LabelEncoder()
    cat_cols = data.select_dtypes(include="object").columns
    for col in cat_cols:
        data[col] = le.fit_transform(data[col].astype(str))

    X = data.drop(columns=["Churn"])
    y = data["Churn"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTrain size : {X_train.shape[0]}  |  Test size : {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, feature_names


# ─────────────────────────────────────────────
# 6. Classical ML Models
# ─────────────────────────────────────────────
def evaluate_model(name: str, y_test, y_pred, y_prob=None) -> dict:
    """Print metrics and return a result dict."""
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    print(f"  Accuracy : {acc:.4f}" + (f"  |  AUC-ROC : {auc:.4f}" if auc else ""))
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    return {"model": name, "accuracy": acc, "auc": auc}


def train_classical_models(
    X_train_scaled, X_test_scaled, y_train, y_test
) -> pd.DataFrame:
    """Train multiple classical classifiers and compare results."""
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    }

    results = []
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else None
        results.append(evaluate_model(name, y_test, y_pred, y_prob))

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)

    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["steelblue" if a < results_df["accuracy"].max() else "tomato"
              for a in results_df["accuracy"]]
    ax.bar(results_df["model"], results_df["accuracy"], color=colors)
    ax.set_ylim(0.70, 0.85)
    ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nModel comparison chart saved → model_comparison.png")

    print("\n" + "=" * 50)
    print("CLASSICAL MODELS SUMMARY")
    print("=" * 50)
    print(results_df.to_string(index=False))
    return results_df


# ─────────────────────────────────────────────
# 7. Feature Importance (Random Forest)
# ─────────────────────────────────────────────
def plot_feature_importance(X_train_scaled, y_train, feature_names: list) -> None:
    """Fit a Random Forest and plot the top-15 feature importances."""
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train_scaled, y_train)

    importances = pd.Series(rf.feature_importances_, index=feature_names)
    top15 = importances.nlargest(15).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    top15.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Feature importance plot saved → feature_importance.png")


# ─────────────────────────────────────────────
# 8. PyTorch Deep Learning Model
# ─────────────────────────────────────────────
class ChurnModel(nn.Module):
    """3-layer feed-forward network with Dropout for binary churn prediction."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_pytorch_model(
    X_train_scaled, X_test_scaled, y_train, y_test
) -> None:
    """Train the PyTorch model and report accuracy."""
    X_tr = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tr = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_te = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_te = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    model = ChurnModel(X_tr.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=PYTORCH_LR)

    print("\n" + "=" * 50)
    print("PYTORCH MODEL TRAINING")
    print("=" * 50)
    losses = []
    for epoch in range(PYTORCH_EPOCHS):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tr)
        loss = criterion(outputs, y_tr)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}  |  Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = (model(X_te) >= 0.5).float()
        acc = (preds == y_te).float().mean().item()
    print(f"\n  PyTorch Test Accuracy : {acc:.4f}")

    # Loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(losses, color="steelblue")
    plt.title("PyTorch Training Loss", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.tight_layout()
    plt.savefig("pytorch_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("PyTorch loss curve saved → pytorch_loss_curve.png")


# ─────────────────────────────────────────────
# 9. Keras / TensorFlow Deep Learning Model
# ─────────────────────────────────────────────
def train_keras_model(
    X_train_scaled, X_test_scaled, y_train, y_test
) -> None:
    """Build, compile, train and evaluate a Keras Sequential model."""
    input_dim = X_train_scaled.shape[1]

    model_keras = keras.Sequential(
        [
            keras.Input(shape=(input_dim,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model_keras.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print("\n" + "=" * 50)
    print("KERAS MODEL TRAINING")
    print("=" * 50)
    history = model_keras.fit(
        X_train_scaled, y_train,
        epochs=KERAS_EPOCHS,
        batch_size=KERAS_BATCH_SIZE,
        validation_data=(X_test_scaled, y_test),
        verbose=1,
    )

    # Accuracy/loss curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Keras Accuracy", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Keras Loss", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("keras_training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Keras training curves saved → keras_training_curves.png")

    loss, acc = model_keras.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\n  Keras Test Accuracy : {acc:.4f}  |  Test Loss : {loss:.4f}")
    print("Training complete!")


# ─────────────────────────────────────────────
# 10. Main Orchestrator
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("   TELCO CUSTOMER CHURN ANALYSIS")
    print("=" * 60)

    # Load data
    df = load_data(FILE_PATH)
    print(f"\nData loaded successfully. Shape: {df.shape}")
    print(df.head())

    # EDA
    run_eda(df)

    # Preprocess
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, feature_names = preprocess(df)

    # Classical ML
    train_classical_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Feature importance
    plot_feature_importance(X_train_scaled, y_train, feature_names)

    # PyTorch deep learning
    train_pytorch_model(X_train_scaled, X_test_scaled, y_train, y_test)

    # Keras deep learning
    train_keras_model(X_train_scaled, X_test_scaled, y_train, y_test)

    print("\n" + "=" * 60)
    print("   ALL STEPS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
