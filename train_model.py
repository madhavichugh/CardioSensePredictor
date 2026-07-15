import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

print("CardioSense AI - Model Training Pipeline")

print("\nLoading dataset")

df = pd.read_csv("cardio_train.csv", sep=";")

print(f"Original Shape: {df.shape}")

if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

df.drop_duplicates(inplace=True)

df["age_years"] = (df["age"] / 365).astype(int)

df = df[(df["ap_hi"] > 60) & (df["ap_hi"] < 240)]
df = df[(df["ap_lo"] > 40) & (df["ap_lo"] < 180)]
df = df[df["ap_hi"] >= df["ap_lo"]]

df = df[(df["height"] > 130) & (df["height"] < 220)]
df = df[(df["weight"] > 30) & (df["weight"] < 200)]

print(f"After Cleaning: {df.shape}")

print("\nCreating Features")

df["BMI"] = (
    df["weight"] /
    ((df["height"] / 100) ** 2)
).round(1)

df["PulsePressure"] = (
    df["ap_hi"] - df["ap_lo"]
)

df["age_group"] = pd.cut(
    df["age_years"],
    bins=[0, 40, 50, 60, 100],
    labels=[0, 1, 2, 3]
).astype(int)

def bp_category(row):
    if row["ap_hi"] < 120 and row["ap_lo"] < 80:
        return 0
    elif row["ap_hi"] < 130 and row["ap_lo"] < 80:
        return 1
    elif row["ap_hi"] < 140 or row["ap_lo"] < 90:
        return 2
    else:
        return 3

df["bp_category"] = df.apply(
    bp_category,
    axis=1
)

df["risk_score"] = (
    df["cholesterol"]
    + df["gluc"]
    + df["smoke"]
    + df["alco"]
    + (1 - df["active"])
)

FEATURE_COLS = [
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "age_years",
    "BMI",
    "PulsePressure",
    "age_group",
    "bp_category",
    "risk_score"
]

X = df[FEATURE_COLS]
y = df["cardio"]

print("\nTrain Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Scaling Features")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nComparing Models")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

results = []

best_model_name = None
best_model = None
best_accuracy = 0

for name, model in models.items():

    print(f"\nTraining {name}")

    model.fit(
        X_train_scaled,
        y_train
    )

    y_pred = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_model = model

print("\n")
print("MODEL COMPARISON SUMMARY")

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]
)
print(results_df)

print("\n")
print("CROSS VALIDATION")

for name, model in models.items():
    scores = cross_val_score(
        model,
        X_train_scaled,
        y_train,
        cv=5,
        scoring="accuracy"
    )
    print(
        f"{name:22s} : "
        f"{scores.mean():.4f} (+/- {scores.std():.4f})"
    )

print("\n")
print("HYPERPARAMETER TUNING")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 5],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(
    X_train_scaled,
    y_train
)

best_model = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)

print(
    f"Best Cross Validation Accuracy : "
    f"{grid.best_score_:.4f}"
)

print("\n")
print("FINAL MODEL EVALUATION")

y_pred = best_model.predict(
    X_test_scaled
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report\n")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\n")
print("FEATURE IMPORTANCE")

importance = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Importance": best_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)

print("\nSaving model")

joblib.dump(
    best_model,
    "heart_disease_model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("Model saved as heart_disease_model.pkl")
print("Scaler saved as scaler.pkl")

print("\n")
print("TRAINING COMPLETED")

print(f"Dataset Size       : {len(df)}")
print(f"Training Samples   : {len(X_train)}")
print(f"Testing Samples    : {len(X_test)}")
print(f"Selected Model     : Random Forest")
print(f"Final Accuracy     : {accuracy:.4f}")
print(f"Final Precision    : {precision:.4f}")
print(f"Final Recall       : {recall:.4f}")
print(f"Final F1 Score     : {f1:.4f}")

print("\nCardioSense AI model trained successfully.")

