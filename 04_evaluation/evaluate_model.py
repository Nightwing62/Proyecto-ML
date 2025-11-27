import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

DATA_PATH = ART / "train_merged.parquet"      
MODEL_PATH = ART / "model_lgbm.joblib"        

print("Cargando dataset final...")
df = pd.read_parquet(DATA_PATH)

assert "TARGET" in df.columns, "El dataset no tiene la columna TARGET"

y = df["TARGET"].values

num_cols = df.select_dtypes(include=["number"]).columns.tolist()
num_cols = [c for c in num_cols if c != "TARGET"]
X = df[num_cols]

print(f"Shape X: {X.shape}, y: {y.shape}")

print("Cargando modelo...")
model = joblib.load(MODEL_PATH)

print("Haciendo predicciones...")
y_proba = model.predict(X)         
y_pred = (y_proba >= 0.5).astype(int)

print("Calculando métricas...")
auc = roc_auc_score(y, y_proba)
cm = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)

print(f"\nAUC: {auc:.4f}")
print("\nMatriz de confusión:")
print(cm)
print("\nReporte de clasificación:")
print(report)

METRICS_PATH = ART / "metrics.txt"
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    f.write(f"AUC: {auc:.4f}\n\n")
    f.write("Matriz de confusión:\n")
    f.write(str(cm))
    f.write("\n\nReporte de clasificación:\n")
    f.write(report)

print(f"\n✔ Métricas guardadas en: {METRICS_PATH}")
