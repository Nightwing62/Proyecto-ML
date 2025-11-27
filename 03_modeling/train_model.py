import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import joblib


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

print("Leyendo dataset final...")
df = pd.read_parquet(ART / "train_merged.parquet")

print("Shape del dataset:", df.shape)

assert "TARGET" in df.columns, "El dataset final debe tener la columna TARGET"
y = df["TARGET"]
X = df.drop(columns=["TARGET"])

num_cols = X.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()
print(f"Usando {len(num_cols)} columnas numéricas de {X.shape[1]} totales")
X = X[num_cols]

X = X.fillna(0)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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

print("Entrenando modelo LightGBM...")

callbacks = [
    lgb.early_stopping(stopping_rounds=50),
    lgb.log_evaluation(period=50),
]

model = lgb.train(
    params,
    train_set=train_set,
    num_boost_round=1000,
    valid_sets=[valid_set],
    valid_names=["valid"],
    callbacks=callbacks,
)

y_valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
auc = roc_auc_score(y_valid, y_valid_pred)
print(f"AUC en valid: {auc:.4f}")

model_path = ART / "model_lgbm.joblib"
cols_path = ART / "model_features.txt"

joblib.dump(model, model_path)
with open(cols_path, "w", encoding="utf-8") as f:
    for c in num_cols:
        f.write(c + "\n")

print(f"Modelo guardado en: {model_path}")
print(f"Lista de features guardada en: {cols_path}")
print("Listo ")
