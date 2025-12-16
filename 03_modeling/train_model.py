import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import joblib


# RUTAS
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

print("Leyendo dataset final...")
df = pd.read_parquet(ART / "train_merged.parquet")

print("Shape dataset completo:", df.shape)


# TARGET 

y = df["TARGET"]
X = df.drop(columns=["TARGET"])

num_cols = X.select_dtypes(include=["number", "bool"]).columns
X = X[num_cols].fillna(0)

print("Total features:", X.shape[1])


# SPLIT 1

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,      # 30% queda fuera del entrenamiento
    random_state=42,
    stratify=y
)


# SPLIT 2

X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,      # mitad valid / mitad test
    random_state=42,
    stratify=y_temp
)

print("Train:", X_train.shape)
print("Valid:", X_valid.shape)
print("Test :", X_test.shape)


# GUARDAR TEST SET (FASE 4)

joblib.dump(X_test.index, ART / "test_idx.joblib")
joblib.dump(y_test, ART / "y_test.joblib")


# LIGHTGBM 

train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_valid, label=y_valid)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}

print("Entrenando modelo...")

model = lgb.train(
    params,
    train_set,
    num_boost_round=1000,
    valid_sets=[valid_set],
    valid_names=["valid"],
    callbacks=[lgb.early_stopping(50)]
)


# EARLY STOPPING

y_valid_pred = model.predict(X_valid)
auc_valid = roc_auc_score(y_valid, y_valid_pred)
print(f"AUC VALID: {auc_valid:.4f}")


# GUARDAR MODELO

joblib.dump(model, ART / "model_lgbm.joblib")

with open(ART / "model_features.txt", "w") as f:
    for c in num_cols:
        f.write(c + "\n")

print("FASE 3 completada")
