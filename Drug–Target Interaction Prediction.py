import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Dataset

print("Loading dataset...")

df = pd.read_csv(
    "data/interactions.tsv",
    sep="\t",
    nrows=20000,         
    low_memory=False
)

print("Original shape:", df.shape)

# 2. Select Required Columns

required_columns = [
    "Ligand SMILES",
    "Target Name",
    "IC50 (nM)"
]

df = df[required_columns]

# 3. Data Cleaning

print("\nCleaning data...")

df = df.dropna()
df = df.drop_duplicates()

df["IC50 (nM)"] = pd.to_numeric(df["IC50 (nM)"], errors="coerce")
df = df.dropna()

print("After cleaning:", df.shape)

# 4. Create Binary Interaction Label

df["interaction"] = (df["IC50 (nM)"] <= 1000).astype(int)

print("\nClass distribution:")
print(df["interaction"].value_counts())

# 5. Optional Sampling

if len(df) > 15000:
    df = df.sample(n=15000, random_state=42)

print("After sampling:", df.shape)

# 6. Encode Drug and Target

le_drug = LabelEncoder()
le_target = LabelEncoder()

df["drug_encoded"] = le_drug.fit_transform(df["Ligand SMILES"])
df["target_encoded"] = le_target.fit_transform(df["Target Name"])

# 7. Define Features and Labels

X = df[["drug_encoded", "target_encoded"]]
y = df["interaction"]

# 8. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 9. Train Model

print("\nTraining model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 10. Evaluate Model

print("\nEvaluating model...")

predictions = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

print("\nDone 🚀")
