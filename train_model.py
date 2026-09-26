import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer


# ==========================================
# 1. Load prepared dataset
# ==========================================

df = pd.read_csv("data/prepared_startup_data.csv")

print("=" * 50)
print("STARTUP SUCCESS PREDICTION - MODEL TRAINING")
print("=" * 50)

print("\nDataset shape:")
print(df.shape)


# ==========================================
# 2. Select features
# ==========================================

features = [
    "funding_total_usd",
    "funding_rounds",
    "milestones",
    "relationships",
    "avg_participants"
]

X = df[features]
y = df["success"]

print("\nSelected Features:")
for feature in features:
    print("-", feature)

print("\nTarget: success")


# ==========================================
# 3. Handle missing values
# ==========================================

print("\nHandling missing values...")

imputer = SimpleImputer(strategy="median")

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=features
)

print("Missing values handled successfully!")


# ==========================================
# 4. Split dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 5. Train Random Forest model
# ==========================================

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully!")


# ==========================================
# 6. Evaluate model
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 7. Save model
# ==========================================

print("\nSaving model...")

joblib.dump(
    model,
    "model/startup_success_model.pkl"
)

joblib.dump(
    imputer,
    "model/imputer.pkl"
)

joblib.dump(
    features,
    "model/feature_columns.pkl"
)

print("\nModel saved successfully!")
print("Imputer saved successfully!")
print("Feature columns saved successfully!")

print("\n" + "=" * 50)
print("TRAINING COMPLETED")
print("=" * 50)